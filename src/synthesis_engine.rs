use crate::{
    acoustic_feature_extractor::OjtPhoneme,
    full_context_label::{Phoneme, Utterance},
    inference::{decode, predict_duration, predict_intonation},
    mora_list::MORA_LIST_MINIMUM,
};
use anyhow::Result;
use ort::Session;

const UNVOICED_MORA_PHONEME_LIST: &[&str] = &["A", "I", "U", "E", "O", "cl", "pau"];
const MORA_PHONEME_LIST: &[&str] = &[
    "a", "i", "u", "e", "o", "N", "A", "I", "U", "E", "O", "cl", "pau",
];

#[derive(Clone)]
struct MoraModel {
    text: String,
    consonant: Option<String>,
    consonant_length: Option<f32>,
    vowel: String,
    vowel_length: f32,
    pitch: f32,
}

#[derive(Clone)]
pub struct AccentPhraseModel {
    moras: Vec<MoraModel>,
    accent: usize,
    pause_mora: Option<MoraModel>,
    is_interrogative: bool,
}

pub fn create_accent_phrases(labels: Vec<String>) -> Result<Vec<AccentPhraseModel>> {
    let utterance = Utterance::from_phonemes(
        labels
            .into_iter()
            .map(|x| Phoneme::from_label(x).unwrap())
            .collect(),
    )?;

    let accent_phrases: Vec<AccentPhraseModel> = utterance.breath_groups.iter().enumerate().fold(
        Vec::new(),
        |mut accum_vec, (i, breath_group)| {
            accum_vec.extend(breath_group.accent_phrases.iter().enumerate().map(
                |(j, accent_phrase)| {
                    let moras = accent_phrase
                        .moras
                        .iter()
                        .map(|mora| {
                            let mora_text = mora
                                .phonemes()
                                .iter()
                                .map(|phoneme| phoneme.phoneme().to_string())
                                .collect::<Vec<_>>()
                                .join("");

                            let (consonant, consonant_length) =
                                if let Some(consonant) = mora.consonant.clone() {
                                    (Some(consonant.phoneme().to_string()), Some(0.))
                                } else {
                                    (None, None)
                                };

                            MoraModel {
                                text: mora_to_text(mora_text),
                                consonant,
                                consonant_length,
                                vowel: mora.vowel.phoneme().into(),
                                vowel_length: 0.,
                                pitch: 0.,
                            }
                        })
                        .collect();

                    let pause_mora = if i != utterance.breath_groups.len() - 1
                        && j == breath_group.accent_phrases.len() - 1
                    {
                        Some(MoraModel {
                            text: "、".into(),
                            consonant: None,
                            consonant_length: None,
                            vowel: "pau".into(),
                            vowel_length: 0.,
                            pitch: 0.,
                        })
                    } else {
                        None
                    };

                    AccentPhraseModel {
                        moras,
                        accent: accent_phrase.accent,
                        pause_mora,
                        is_interrogative: accent_phrase.is_interrogative,
                    }
                },
            ));

            accum_vec
        },
    );

    Ok(accent_phrases)
}

pub fn replace_phoneme_length(
    session: Session,
    accent_phrases: Vec<AccentPhraseModel>,
    speaker_id: u32,
) -> Result<Vec<AccentPhraseModel>> {
    let (_, phoneme_data_list) = initial_process(accent_phrases.clone());
    let (_, _, vowel_indexes_data) = split_mora(phoneme_data_list.clone());

    let phoneme_list_s: Vec<i64> = phoneme_data_list
        .iter()
        .map(OjtPhoneme::phoneme_id)
        .collect();
    let phoneme_length = predict_duration(session, &phoneme_list_s, speaker_id)?;

    let mut index = 0;
    let new_accent_phrases = accent_phrases
        .into_iter()
        .map(|accent_phrase| AccentPhraseModel {
            moras: accent_phrase
                .moras
                .into_iter()
                .map(|mora| {
                    let new_mora = MoraModel {
                        text: mora.text,
                        consonant: mora.consonant.clone(),
                        consonant_length: mora
                            .consonant
                            .map(|_| phoneme_length[vowel_indexes_data[index + 1] as usize - 1]),
                        vowel: mora.vowel.clone(),
                        vowel_length: phoneme_length[vowel_indexes_data[index + 1] as usize],
                        pitch: mora.pitch,
                    };
                    index += 1;
                    new_mora
                })
                .collect(),
            accent: accent_phrase.accent,
            pause_mora: accent_phrase.pause_mora.map(|pause_mora| {
                let new_pause_mora = MoraModel {
                    text: pause_mora.text,
                    consonant: pause_mora.consonant,
                    consonant_length: pause_mora.consonant_length,
                    vowel: pause_mora.vowel,
                    vowel_length: phoneme_length[vowel_indexes_data[index + 1] as usize],
                    pitch: pause_mora.pitch,
                };
                index += 1;
                new_pause_mora
            }),
            is_interrogative: accent_phrase.is_interrogative,
        })
        .collect();

    Ok(new_accent_phrases)
}

pub fn replace_mora_pitch(
    session: Session,
    accent_phrases: Vec<AccentPhraseModel>,
    speaker_id: u32,
) -> Result<Vec<AccentPhraseModel>> {
    let (_, phoneme_data_list) = initial_process(accent_phrases.clone());
    let (consonant_phoneme_data_list, vowel_phoneme_data_list, vowel_indexes) =
        split_mora(phoneme_data_list);
    let consonant_phoneme_list: Vec<i64> = consonant_phoneme_data_list
        .iter()
        .map(OjtPhoneme::phoneme_id)
        .collect();
    let vowel_phoneme_list: Vec<i64> = vowel_phoneme_data_list
        .iter()
        .map(OjtPhoneme::phoneme_id)
        .collect();

    let base_start_accent_list: Vec<i64> = std::iter::once(0)
        .chain(accent_phrases.iter().flat_map(|accent_phrase| {
            create_one_accent_list(accent_phrase, (accent_phrase.accent != 1) as i32)
        }))
        .chain(std::iter::once(0))
        .collect();
    let base_end_accent_list: Vec<i64> = std::iter::once(0)
        .chain(accent_phrases.iter().flat_map(|accent_phrase| {
            create_one_accent_list(accent_phrase, accent_phrase.accent as i32 - 1)
        }))
        .chain(std::iter::once(0))
        .collect();
    let base_start_accent_phrase_list: Vec<i64> = std::iter::once(0)
        .chain(
            accent_phrases
                .iter()
                .flat_map(|accent_phrase| create_one_accent_list(accent_phrase, 0)),
        )
        .chain(std::iter::once(0))
        .collect();
    let base_end_accent_phrase_list: Vec<i64> = std::iter::once(0)
        .chain(
            accent_phrases
                .iter()
                .flat_map(|accent_phrase| create_one_accent_list(accent_phrase, -1)),
        )
        .chain(std::iter::once(0))
        .collect();

    let start_accent_list: Vec<i64> = vowel_indexes
        .iter()
        .map(|vowel_index| base_start_accent_list[*vowel_index as usize])
        .collect();
    let end_accent_list: Vec<i64> = vowel_indexes
        .iter()
        .map(|vowel_index| base_end_accent_list[*vowel_index as usize])
        .collect();
    let start_accent_phrase_list: Vec<i64> = vowel_indexes
        .iter()
        .map(|vowel_index| base_start_accent_phrase_list[*vowel_index as usize])
        .collect();
    let end_accent_phrase_list: Vec<i64> = vowel_indexes
        .iter()
        .map(|vowel_index| base_end_accent_phrase_list[*vowel_index as usize])
        .collect();

    let f0_list: Vec<f32> = predict_intonation(
        session,
        vowel_phoneme_list.len(),
        &vowel_phoneme_list,
        &consonant_phoneme_list,
        &start_accent_list,
        &end_accent_list,
        &start_accent_phrase_list,
        &end_accent_phrase_list,
        speaker_id,
    )?
    .into_iter()
    .enumerate()
    .map(|(i, f0)| {
        if UNVOICED_MORA_PHONEME_LIST
            .iter()
            .any(|phoneme| *phoneme == vowel_phoneme_data_list[i].phoneme)
        {
            0.
        } else {
            f0
        }
    })
    .collect();

    let mut index = 0;
    let new_accent_phrases = accent_phrases
        .into_iter()
        .map(|accent_phrase| AccentPhraseModel {
            moras: accent_phrase
                .moras
                .into_iter()
                .map(|mora| {
                    let new_mora = MoraModel {
                        text: mora.text,
                        consonant: mora.consonant,
                        consonant_length: mora.consonant_length,
                        vowel: mora.vowel,
                        vowel_length: mora.vowel_length,
                        pitch: f0_list[index + 1],
                    };
                    index += 1;
                    new_mora
                })
                .collect(),
            accent: accent_phrase.accent,
            pause_mora: accent_phrase.pause_mora.map(|pause_mora| {
                let new_pause_mora = MoraModel {
                    text: pause_mora.text,
                    consonant: pause_mora.consonant,
                    consonant_length: pause_mora.consonant_length,
                    vowel: pause_mora.vowel,
                    vowel_length: pause_mora.vowel_length,
                    pitch: f0_list[index + 1],
                };
                index += 1;
                new_pause_mora
            }),
            is_interrogative: accent_phrase.is_interrogative,
        })
        .collect();

    Ok(new_accent_phrases)
}

pub fn synthesis(
    session: Session,
    accent_phrases: Vec<AccentPhraseModel>,
    speed_scale: f32,
    pitch_scale: f32,
    intonation_scale: f32,
    pre_phoneme_length: f32,
    post_phoneme_length: f32,
    enable_interrogative_upspeak: bool,
    speaker_id: u32,
) -> Result<Vec<f32>> {
    let accent_phrases = if enable_interrogative_upspeak {
        adjust_interrogative_accent_phrases(accent_phrases)
    } else {
        accent_phrases
    };

    let (flatten_moras, phoneme_data_list) = initial_process(accent_phrases);

    let mut phoneme_length_list = vec![pre_phoneme_length];
    let mut f0_list = vec![0.];
    let mut voiced_list = vec![false];
    {
        let mut sum_of_f0_bigger_than_zero = 0.;
        let mut count_of_f0_bigger_than_zero = 0;

        for mora in flatten_moras {
            let consonant_length = mora.consonant_length;
            let vowel_length = mora.vowel_length;
            let pitch = mora.pitch;

            if let Some(consonant_length) = consonant_length {
                phoneme_length_list.push(consonant_length);
            }
            phoneme_length_list.push(vowel_length);

            let f0_single = pitch * 2.0_f32.powf(pitch_scale);
            f0_list.push(f0_single);

            let bigger_than_zero = f0_single > 0.;
            voiced_list.push(bigger_than_zero);

            if bigger_than_zero {
                sum_of_f0_bigger_than_zero += f0_single;
                count_of_f0_bigger_than_zero += 1;
            }
        }
        phoneme_length_list.push(post_phoneme_length);
        f0_list.push(0.);
        voiced_list.push(false);
        let mean_f0 = sum_of_f0_bigger_than_zero / (count_of_f0_bigger_than_zero as f32);

        if !mean_f0.is_nan() {
            for i in 0..f0_list.len() {
                if voiced_list[i] {
                    f0_list[i] = (f0_list[i] - mean_f0) * intonation_scale + mean_f0;
                }
            }
        }
    }

    let (_, _, vowel_indexes) = split_mora(phoneme_data_list.clone());

    let mut phoneme: Vec<Vec<f32>> = Vec::new();
    let mut f0: Vec<f32> = Vec::new();
    {
        const RATE: f32 = 24000. / 256.;
        let mut sum_of_phoneme_length = 0;
        let mut count_of_f0 = 0;
        let mut vowel_indexes_index = 0;

        for (i, phoneme_length) in phoneme_length_list.iter().enumerate() {
            let phoneme_length = (*phoneme_length * RATE / speed_scale).ceil() as usize;
            let phoneme_id = phoneme_data_list[i].phoneme_id();

            for _ in 0..phoneme_length {
                let mut phonemes_vec = vec![0.; OjtPhoneme::num_phoneme()];
                phonemes_vec[phoneme_id as usize] = 1.;
                phoneme.push(phonemes_vec)
            }
            sum_of_phoneme_length += phoneme_length;

            if i as i64 == vowel_indexes[vowel_indexes_index] {
                for _ in 0..sum_of_phoneme_length {
                    f0.push(f0_list[count_of_f0]);
                }
                count_of_f0 += 1;
                sum_of_phoneme_length = 0;
                vowel_indexes_index += 1;
            }
        }
    }

    // 2次元のvectorを1次元に変換し、アドレスを連続させる
    let flatten_phoneme = phoneme.into_iter().flatten().collect::<Vec<_>>();

    decode(
        session,
        f0.len(),
        OjtPhoneme::num_phoneme(),
        f0,
        flatten_phoneme,
        speaker_id,
    )
}

fn initial_process(accent_phrases: Vec<AccentPhraseModel>) -> (Vec<MoraModel>, Vec<OjtPhoneme>) {
    // to_flatten_moras
    let flatten_moras: Vec<MoraModel> = accent_phrases
        .into_iter()
        .flat_map(|accent_phrase| {
            accent_phrase
                .moras
                .into_iter()
                .chain(accent_phrase.pause_mora.into_iter())
        })
        .collect();

    let phoneme_strings: Vec<String> = std::iter::once("pau".to_string())
        .chain(flatten_moras.clone().into_iter().flat_map(|mora| {
            mora.consonant
                .into_iter()
                .chain(std::iter::once(mora.vowel))
        }))
        .chain(std::iter::once("pau".to_string()))
        .collect();

    // to_phoneme_data_list
    let phoneme_data_list = OjtPhoneme::convert(
        phoneme_strings
            .into_iter()
            .map(|s| OjtPhoneme { phoneme: s })
            .collect(),
    );

    (flatten_moras, phoneme_data_list)
}

fn create_one_accent_list(accent_phrase: &AccentPhraseModel, point: i32) -> Vec<i64> {
    accent_phrase
        .moras
        .iter()
        .enumerate()
        .flat_map(|(i, mora)| {
            let value = (i as i32 == point
                || (point < 0 && i == (accent_phrase.moras.len() as i32 + point) as usize))
                .into();
            std::iter::once(value).chain(mora.consonant.as_ref().map(|_| value))
        })
        .chain(accent_phrase.pause_mora.as_ref().map(|_| 0))
        .collect()
}

fn split_mora(phoneme_list: Vec<OjtPhoneme>) -> (Vec<OjtPhoneme>, Vec<OjtPhoneme>, Vec<i64>) {
    let vowel_indexes: Vec<i64> = phoneme_list
        .iter()
        .enumerate()
        .filter_map(|(i, phoneme)| {
            if MORA_PHONEME_LIST
                .iter()
                .any(|mora_phoneme| *mora_phoneme == phoneme.phoneme)
            {
                Some(i as i64)
            } else {
                None
            }
        })
        .collect();

    let vowel_phoneme_list = vowel_indexes
        .iter()
        .map(|vowel_index| phoneme_list[*vowel_index as usize].clone())
        .collect();

    let consonant_phoneme_list = std::iter::once(OjtPhoneme {
        phoneme: String::new(),
    })
    .chain(vowel_indexes.windows(2).map(|w| {
        let (prev, next) = (w[0], w[1]);
        if next - prev == 1 {
            OjtPhoneme {
                phoneme: String::new(),
            }
        } else {
            phoneme_list[next as usize - 1].clone()
        }
    }))
    .collect();

    (consonant_phoneme_list, vowel_phoneme_list, vowel_indexes)
}

fn mora_to_text(mora: String) -> String {
    // 末尾文字の置換
    let last_char = mora.chars().last().unwrap();
    let mora = if ['A', 'I', 'U', 'E', 'O'].contains(&last_char) {
        format!("{}{}", &mora[0..mora.len() - 1], last_char.to_lowercase())
    } else {
        mora
    };

    // もしカタカナに変換できなければ、引数で与えた文字列がそのまま返ってくる
    MORA_LIST_MINIMUM
        .iter()
        .find_map(|[text, consonant, vowel]| {
            if mora.starts_with(consonant) && mora.ends_with(vowel) {
                Some(text.to_string())
            } else {
                None
            }
        })
        .unwrap_or(mora)
}

fn adjust_interrogative_accent_phrases(
    accent_phrases: Vec<AccentPhraseModel>,
) -> Vec<AccentPhraseModel> {
    accent_phrases
        .into_iter()
        .map(|accent_phrase| {
            let is_interrogative = accent_phrase.is_interrogative;
            let accent = accent_phrase.accent;
            let pause_mora = accent_phrase.pause_mora.clone();
            AccentPhraseModel {
                moras: adjust_interrogative_moras(accent_phrase),
                accent,
                pause_mora,
                is_interrogative,
            }
        })
        .collect()
}

fn adjust_interrogative_moras(accent_phrase: AccentPhraseModel) -> Vec<MoraModel> {
    let moras = accent_phrase.moras;
    if accent_phrase.is_interrogative && !moras.is_empty() {
        let last_mora = moras.last().unwrap().clone();
        let last_mora_pitch = last_mora.pitch;
        if last_mora_pitch != 0.0 {
            let new_moras: Vec<MoraModel> = moras
                .into_iter()
                .chain(std::iter::once(make_interrogative_mora(last_mora)))
                .collect();
            return new_moras;
        }
    }
    moras
}

fn make_interrogative_mora(last_mora: MoraModel) -> MoraModel {
    const FIX_VOWEL_LENGTH: f32 = 0.15;
    const ADJUST_PITCH: f32 = 0.3;
    const MAX_PITCH: f32 = 6.5;

    let pitch = (last_mora.pitch + ADJUST_PITCH).min(MAX_PITCH);

    MoraModel {
        text: mora_to_text(last_mora.vowel.clone()),
        consonant: None,
        consonant_length: None,
        vowel: last_mora.vowel,
        vowel_length: FIX_VOWEL_LENGTH,
        pitch,
    }
}
