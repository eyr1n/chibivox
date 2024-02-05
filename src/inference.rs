use anyhow::Result;
use ort::Session;

const PHONEME_LENGTH_MINIMAL: f32 = 0.01;

pub fn predict_duration(
    session: Session,
    phoneme_vector: &[i64],
    speaker_id: u32,
) -> Result<Vec<f32>> {
    let input_tensors = ort::inputs![
        "phoneme_list" => ndarray::arr1(phoneme_vector),
        "speaker_id" => ndarray::arr1(&[speaker_id as i64])
    ]?;
    let output_tensors = session.run(input_tensors)?;
    let output = output_tensors["phoneme_length"]
        .extract_tensor::<f32>()?
        .view()
        .iter()
        .map(|output_item| {
            if *output_item < PHONEME_LENGTH_MINIMAL {
                PHONEME_LENGTH_MINIMAL
            } else {
                *output_item
            }
        })
        .collect();

    Ok(output)
}

pub fn predict_intonation(
    session: Session,
    length: usize,
    vowel_phoneme_vector: &[i64],
    consonant_phoneme_vector: &[i64],
    start_accent_vector: &[i64],
    end_accent_vector: &[i64],
    start_accent_phrase_vector: &[i64],
    end_accent_phrase_vector: &[i64],
    speaker_id: u32,
) -> Result<Vec<f32>> {
    let input_tensors = ort::inputs![
        "length" => ndarray::arr0(length as i64),
        "vowel_phoneme_list" => ndarray::arr1(vowel_phoneme_vector),
        "consonant_phoneme_list" => ndarray::arr1(consonant_phoneme_vector),
        "start_accent_list" => ndarray::arr1(start_accent_vector),
        "end_accent_list" => ndarray::arr1(end_accent_vector),
        "start_accent_phrase_list" => ndarray::arr1(start_accent_phrase_vector),
        "end_accent_phrase_list" => ndarray::arr1(end_accent_phrase_vector),
        "speaker_id" => ndarray::arr1(&[speaker_id as i64]),
    ]?;
    let output_tensors = session.run(input_tensors)?;
    let output = output_tensors["f0_list"]
        .extract_tensor::<f32>()?
        .view()
        .to_owned()
        .into_raw_vec();

    Ok(output)
}

pub fn decode(
    session: Session,
    length: usize,
    phoneme_size: usize,
    f0: Vec<f32>,
    phoneme_vector: Vec<f32>,
    speaker_id: u32,
) -> Result<Vec<f32>> {
    const PADDING_SIZE: f64 = 0.4;
    const DEFAULT_SAMPLING_RATE: f64 = 24000.0;

    let padding_size = ((PADDING_SIZE * DEFAULT_SAMPLING_RATE) / 256.0).round() as usize;
    let start_and_end_padding_size = 2 * padding_size;
    let length_with_padding = length + start_and_end_padding_size;
    let f0_with_padding = make_f0_with_padding(f0.to_vec(), padding_size);
    let phoneme_with_padding =
        make_phoneme_with_padding(phoneme_vector, phoneme_size, padding_size);

    let input_tensors = ort::inputs![
        "f0" => ndarray::arr1(&f0_with_padding).into_shape([length_with_padding, 1])?,
        "phoneme" => ndarray::arr1(&phoneme_with_padding).into_shape([length_with_padding, phoneme_size])?,
        "speaker_id" => ndarray::arr1(&[speaker_id as i64])
    ]?;
    let output_tensors = session.run(input_tensors)?;
    let output = output_tensors["wave"]
        .extract_tensor::<f32>()?
        .view()
        .to_owned()
        .into_raw_vec();

    Ok(trim_padding_from_output(output, padding_size))
}

fn make_f0_with_padding(f0: Vec<f32>, padding_size: usize) -> Vec<f32> {
    std::iter::repeat(0.0)
        .take(padding_size)
        .chain(f0)
        .chain(std::iter::repeat(0.0).take(padding_size))
        .collect()
}

fn make_phoneme_with_padding(
    phoneme: Vec<f32>,
    phoneme_size: usize,
    padding_size: usize,
) -> Vec<f32> {
    let padding_phonemes = std::iter::once(1.0)
        .chain(std::iter::repeat(0.0).take(phoneme_size - 1))
        .cycle()
        .take(phoneme_size * padding_size);
    padding_phonemes
        .clone()
        .chain(phoneme)
        .chain(padding_phonemes)
        .collect()
}

fn trim_padding_from_output(mut output: Vec<f32>, padding_f0_size: usize) -> Vec<f32> {
    let padding_sampling_size = padding_f0_size * 256;
    output
        .drain(padding_sampling_size..output.len() - padding_sampling_size)
        .collect()
}
