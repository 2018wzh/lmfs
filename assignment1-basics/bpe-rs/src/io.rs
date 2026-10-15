use crate::{core::Engine, lexer::Specials};
use npyz::WriterBuilder;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read},
};

pub fn read_chunks(
    path: &str,
    specials: &Specials,
    block_bytes: usize,
    mut visit: impl FnMut(&str) -> Result<(), String>,
) -> Result<(), String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let src: Box<dyn Read> = if path.ends_with(".gz") {
        Box::new(flate2::read::MultiGzDecoder::new(file))
    } else {
        Box::new(file)
    };
    let mut reader = BufReader::with_capacity(1 << 20, src);
    let mut block = vec![0u8; block_bytes];
    let mut pending = Vec::new();
    loop {
        let n = reader.read(&mut block).map_err(|e| e.to_string())?;
        pending.extend_from_slice(&block[..n]);
        let valid = match std::str::from_utf8(&pending) {
            Ok(s) => s.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(e) => return Err(format!("invalid UTF-8: {e}")),
        };
        let text = std::str::from_utf8(&pending[..valid]).unwrap();
        let cut = if n == 0 {
            text.len()
        } else {
            specials.stable_cut(text)
        };
        if cut > 0 {
            visit(&text[..cut])?;
            pending.drain(..cut);
        }
        if n == 0 {
            return Ok(());
        }
    }
}

pub fn encode_file(
    engine: &Engine,
    input: &str,
    output: &str,
    block_bytes: usize,
) -> Result<u64, String> {
    let file = File::create(output).map_err(|e| e.to_string())?;
    let mut writer = npyz::WriteOptions::<u16>::new()
        .default_dtype()
        .writer(BufWriter::new(file))
        .begin_1d()
        .map_err(|e| e.to_string())?;
    let mut count = 0u64;
    read_chunks(input, &engine.model.specials, block_bytes, |text| {
        let ids = engine.encode(text);
        count += ids.len() as u64;
        writer.extend(ids).map_err(|e| e.to_string())
    })?;
    writer.finish().map_err(|e| e.to_string())?;
    Ok(count)
}
