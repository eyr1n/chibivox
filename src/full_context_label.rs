use anyhow::{anyhow, Context, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;

#[derive(Clone)]
pub struct Phoneme {
    contexts: HashMap<String, String>,
}

static P3_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\-(.*?)\+)").unwrap());
static A2_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\+(\d+|xx)\+)").unwrap());
static A3_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\+(\d+|xx)/B:)").unwrap());
static F1_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(/F:(\d+|xx)_)").unwrap());
static F2_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(_(\d+|xx)\#)").unwrap());
static F3_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\#(\d+|xx)_)").unwrap());
static F5_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(@(\d+|xx)_)").unwrap());
static H1_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(/H:(\d+|xx)_)").unwrap());
static I3_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(@(\d+|xx)\+)").unwrap());
static J1_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(/J:(\d+|xx)_)").unwrap());

fn string_feature_by_regex(re: &Regex, label: &str) -> Result<String> {
    re.captures(label)
        .ok_or(anyhow!("FullContextLabelError::LabelParse"))
        .map(|caps| caps.get(2).unwrap().as_str().to_string())
}

impl Phoneme {
    pub fn from_label(label: impl Into<String>) -> Result<Self> {
        let mut contexts = HashMap::<String, String>::with_capacity(10);
        let label = label.into();
        contexts.insert("p3".into(), string_feature_by_regex(&P3_REGEX, &label)?);
        contexts.insert("a2".into(), string_feature_by_regex(&A2_REGEX, &label)?);
        contexts.insert("a3".into(), string_feature_by_regex(&A3_REGEX, &label)?);
        contexts.insert("f1".into(), string_feature_by_regex(&F1_REGEX, &label)?);
        contexts.insert("f2".into(), string_feature_by_regex(&F2_REGEX, &label)?);
        contexts.insert("f3".into(), string_feature_by_regex(&F3_REGEX, &label)?);
        contexts.insert("f5".into(), string_feature_by_regex(&F5_REGEX, &label)?);
        contexts.insert("h1".into(), string_feature_by_regex(&H1_REGEX, &label)?);
        contexts.insert("i3".into(), string_feature_by_regex(&I3_REGEX, &label)?);
        contexts.insert("j1".into(), string_feature_by_regex(&J1_REGEX, &label)?);

        Ok(Self { contexts })
    }

    pub fn phoneme(&self) -> &str {
        self.contexts.get("p3").unwrap().as_str()
    }

    fn is_pause(&self) -> bool {
        self.contexts.get("f1").unwrap().as_str() == "xx"
    }
}

#[derive(Clone)]
pub struct Mora {
    pub consonant: Option<Phoneme>,
    pub vowel: Phoneme,
}

impl Mora {
    pub fn phonemes(&self) -> Vec<Phoneme> {
        if self.consonant.is_some() {
            vec![self.consonant.as_ref().unwrap().clone(), self.vowel.clone()]
        } else {
            vec![self.vowel.clone()]
        }
    }
}

#[derive(Clone)]
pub struct AccentPhrase {
    pub moras: Vec<Mora>,
    pub accent: usize,
    pub is_interrogative: bool,
}

impl AccentPhrase {
    pub fn from_phonemes(mut phonemes: Vec<Phoneme>) -> Result<Self> {
        let mut moras = Vec::with_capacity(phonemes.len());
        let mut mora_phonemes = Vec::with_capacity(phonemes.len());
        for i in 0..phonemes.len() {
            {
                let phoneme = phonemes.get_mut(i).unwrap();
                if phoneme.contexts.get("a2").map(|s| s.as_str()) == Some("49") {
                    break;
                }
                mora_phonemes.push(phoneme.clone());
            }

            if i + 1 == phonemes.len()
                || phonemes.get(i).unwrap().contexts.get("a2").unwrap()
                    != phonemes.get(i + 1).unwrap().contexts.get("a2").unwrap()
            {
                if mora_phonemes.len() == 1 {
                    moras.push(Mora {
                        consonant: None,
                        vowel: mora_phonemes.get(0).unwrap().clone(),
                    });
                } else if mora_phonemes.len() == 2 {
                    moras.push(Mora {
                        consonant: Some(mora_phonemes.get(0).unwrap().clone()),
                        vowel: mora_phonemes.get(1).unwrap().clone(),
                    });
                } else {
                    return Err(anyhow!("FullContextLabelError::TooLongMora"));
                }
                mora_phonemes.clear();
            }
        }

        let mora = moras.get(0).unwrap();
        let mut accent: usize = mora
            .vowel
            .contexts
            .get("f2")
            .context("FullContextLabelError::InvalidMora")?
            .parse()?;

        let is_interrogative = moras
            .last()
            .unwrap()
            .vowel
            .contexts
            .get("f3")
            .map(|s| s.as_str())
            == Some("1");
        // workaround for VOICEVOX/voicevox_engine#55
        if accent > moras.len() {
            accent = moras.len();
        }

        Ok(Self {
            moras,
            accent,
            is_interrogative,
        })
    }
}

#[derive(Clone)]
pub struct BreathGroup {
    pub accent_phrases: Vec<AccentPhrase>,
}

impl BreathGroup {
    pub fn from_phonemes(phonemes: Vec<Phoneme>) -> Result<Self> {
        let mut accent_phrases = Vec::with_capacity(phonemes.len());
        let mut accent_phonemes = Vec::with_capacity(phonemes.len());
        for i in 0..phonemes.len() {
            accent_phonemes.push(phonemes.get(i).unwrap().clone());
            if i + 1 == phonemes.len()
                || phonemes.get(i).unwrap().contexts.get("i3").unwrap()
                    != phonemes.get(i + 1).unwrap().contexts.get("i3").unwrap()
                || phonemes.get(i).unwrap().contexts.get("f5").unwrap()
                    != phonemes.get(i + 1).unwrap().contexts.get("f5").unwrap()
            {
                accent_phrases.push(AccentPhrase::from_phonemes(accent_phonemes.clone())?);
                accent_phonemes.clear();
            }
        }

        Ok(Self { accent_phrases })
    }
}

#[derive(Clone)]
pub struct Utterance {
    pub breath_groups: Vec<BreathGroup>,
}

impl Utterance {
    pub fn from_phonemes(phonemes: Vec<Phoneme>) -> Result<Self> {
        let mut breath_groups = vec![];
        let mut group_phonemes = Vec::with_capacity(phonemes.len());
        for phoneme in phonemes.into_iter() {
            if !phoneme.is_pause() {
                group_phonemes.push(phoneme);
            } else {
                if !group_phonemes.is_empty() {
                    breath_groups.push(BreathGroup::from_phonemes(group_phonemes.clone())?);
                    group_phonemes.clear();
                }
            }
        }
        Ok(Self { breath_groups })
    }
}
