mod acoustic_feature_extractor;
mod full_context_label;
mod inference;
mod mora_list;
mod synthesis_engine;

use anyhow::{anyhow, Result};
use jpreprocess::{
    kind::JPreprocessDictionaryKind, JPreprocess, JPreprocessConfig, SystemDictionaryConfig,
};
use ort::Session;
use std::fs::File;

const SAMPLING_RATE: u32 = 24000;

fn main() -> Result<()> {
    let text = std::env::args().nth(1).ok_or(anyhow!("invalid args"))?;

    // JPreprocess
    let config = JPreprocessConfig {
        dictionary: SystemDictionaryConfig::Bundled(JPreprocessDictionaryKind::NaistJdic),
        user_dictionary: None,
    };
    let jpreprocess = JPreprocess::from_config(config)?;
    let labels = jpreprocess.extract_fullcontext(text.as_ref())?;

    // Session生成
    let predict_duration =
        Session::builder()?.with_model_from_file("model/predict_duration-0.onnx")?;
    let predict_intonation =
        Session::builder()?.with_model_from_file("model/predict_intonation-0.onnx")?;
    let decode = Session::builder()?.with_model_from_file("model/decode-0.onnx")?;

    // AudioQuery生成
    let accent_phrases = synthesis_engine::create_accent_phrases(labels)?;
    let accent_phrases =
        synthesis_engine::replace_phoneme_length(predict_duration, accent_phrases, 0)?;
    let accent_phrases =
        synthesis_engine::replace_mora_pitch(predict_intonation, accent_phrases, 0)?;

    // 合成
    let wav = synthesis_engine::synthesis(decode, accent_phrases, 1., 0., 1., 0.1, 0.1, true, 0)?;

    // 保存
    let head = wav_io::new_header(SAMPLING_RATE, 32, true, true);
    let mut file = File::create("audio.wav")?;
    wav_io::write_to_file(&mut file, &head, &wav).map_err(|_| anyhow!("wav output error"))?;

    Ok(())
}
