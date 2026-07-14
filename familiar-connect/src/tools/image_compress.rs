//! JPEG compression for images fetched by the `view_image` tool (subsystem 08;
//! Python `tools/image_compress.py`; feature `images`).
//!
//! Resizes to a 1024px longest edge (downscale only), then steps JPEG quality
//! down until the output fits under a byte ceiling. Animated GIFs are tiled into
//! a horizontal strip of up to four evenly-spaced frames so the vision model
//! sees the animation arc. Output byte sizes differ from Pillow (no `optimize`
//! equivalent); the conformance target is the ceiling / dimensions, not bytes
//! (DESIGN / spec 08 port notes).

use std::io::Cursor;

use base64::Engine as _;
use image::codecs::gif::GifDecoder;
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{
    AnimationDecoder, DynamicImage, ExtendedColorType, ImageEncoder, ImageFormat, RgbImage,
    RgbaImage,
};

const MAX_EDGE: u32 = 1024;
const START_QUALITY: u8 = 85;
const START_QUALITY_DESCRIBE: u8 = 95;
const MIN_QUALITY: u8 = 20;
const QUALITY_STEP: u8 = 5;
/// Prose-payload byte ceiling.
pub const SIZE_CEILING: usize = 1_000_000;
const SIZE_CEILING_DESCRIBE: usize = 4_000_000;
/// Max frames tiled from an animated GIF.
pub const MAX_GIF_FRAMES: usize = 4;

/// Compression / decode error.
#[derive(Debug, thiserror::Error)]
pub enum ImageCompressError {
    /// No quality setting achieves the target size (Python `ImageTooLargeError`).
    #[error("image cannot be compressed under {0} bytes")]
    TooLarge(usize),
    /// Underlying decode/encode failure.
    #[error("image codec error: {0}")]
    Codec(#[from] image::ImageError),
}

fn rgba_to_rgb(rgba: &RgbaImage) -> RgbImage {
    DynamicImage::ImageRgba8(rgba.clone()).to_rgb8()
}

/// Open image bytes as RGB; tile evenly-spaced frames for animated GIFs.
///
/// Animated GIFs produce a horizontal strip of up to [`MAX_GIF_FRAMES`]
/// evenly-spaced frames scaled to the tallest frame's height. Static images
/// (including single-frame GIFs) convert to RGB directly.
///
/// # Errors
/// Propagates a decode failure.
pub fn open_as_rgb(raw: &[u8]) -> Result<RgbImage, ImageCompressError> {
    let is_gif = image::guess_format(raw).is_ok_and(|f| f == ImageFormat::Gif);
    if !is_gif {
        return Ok(image::load_from_memory(raw)?.to_rgb8());
    }

    let decoder = GifDecoder::new(Cursor::new(raw))?;
    let frames = decoder.into_frames().collect_frames()?;
    let n = frames.len();
    if n <= 1 {
        let rgba = frames
            .into_iter()
            .next()
            .expect("gif has at least one frame")
            .into_buffer();
        return Ok(rgba_to_rgb(&rgba));
    }

    let count = MAX_GIF_FRAMES.min(n);
    // round(i*(n-1) / (count-1)) via integer half-up. For these inputs no exact
    // .5 tie occurs, so this equals Python's banker's-rounding result.
    let indices: Vec<usize> = (0..count)
        .map(|i| (2 * i * (n - 1) + (count - 1)) / (2 * (count - 1)))
        .collect();

    let selected: Vec<RgbImage> = indices
        .iter()
        .map(|&idx| rgba_to_rgb(frames[idx].buffer()))
        .collect();
    let target_h = selected.iter().map(RgbImage::height).max().unwrap_or(1);
    let parts: Vec<RgbImage> = selected
        .into_iter()
        .map(|f| {
            if f.height() == target_h {
                f
            } else {
                let nw = u32::try_from(
                    u64::from(f.width()) * u64::from(target_h) / u64::from(f.height()),
                )
                .unwrap_or(u32::MAX)
                .max(1);
                image::imageops::resize(&f, nw, target_h, FilterType::Lanczos3)
            }
        })
        .collect();

    let total_w: u32 = parts.iter().map(RgbImage::width).sum();
    let mut composite = RgbImage::new(total_w, target_h);
    let mut x = 0i64;
    for p in &parts {
        image::imageops::overlay(&mut composite, p, x, 0);
        x += i64::from(p.width());
    }
    Ok(composite)
}

fn downscale(img: RgbImage, max_edge: u32) -> RgbImage {
    if img.width() <= max_edge && img.height() <= max_edge {
        return img;
    }
    DynamicImage::ImageRgb8(img)
        .resize(max_edge, max_edge, FilterType::Lanczos3)
        .to_rgb8()
}

fn encode_jpeg(rgb: &RgbImage, quality: u8) -> Result<Vec<u8>, image::ImageError> {
    let mut out: Vec<u8> = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut out, quality);
    encoder.write_image(
        rgb.as_raw(),
        rgb.width(),
        rgb.height(),
        ExtendedColorType::Rgb8,
    )?;
    Ok(out)
}

fn compress_ladder(
    rgb: &RgbImage,
    start_quality: u8,
    ceiling: usize,
) -> Result<Vec<u8>, ImageCompressError> {
    let mut quality = start_quality;
    loop {
        let data = encode_jpeg(rgb, quality)?;
        if data.len() <= ceiling {
            return Ok(data);
        }
        if quality <= MIN_QUALITY {
            break;
        }
        quality = quality.saturating_sub(QUALITY_STEP);
    }
    Err(ImageCompressError::TooLarge(ceiling))
}

/// Resize to a 1024px edge and encode as JPEG under `ceiling` bytes.
///
/// # Errors
/// [`ImageCompressError::TooLarge`] if even quality 20 exceeds `ceiling`.
pub fn compress_to_jpeg(raw: &[u8], ceiling: usize) -> Result<Vec<u8>, ImageCompressError> {
    let img = downscale(open_as_rgb(raw)?, MAX_EDGE);
    compress_ladder(&img, START_QUALITY, ceiling)
}

/// High-quality JPEG for vision-model description; 4 MB ceiling, starts at q95.
///
/// # Errors
/// [`ImageCompressError::TooLarge`] if even quality 20 exceeds the ceiling.
pub fn compress_for_description(raw: &[u8]) -> Result<Vec<u8>, ImageCompressError> {
    let img = downscale(open_as_rgb(raw)?, MAX_EDGE);
    compress_ladder(&img, START_QUALITY_DESCRIBE, SIZE_CEILING_DESCRIBE)
}

/// Compress `raw` to JPEG (1 MB ceiling) and base64-encode.
///
/// # Errors
/// Propagates a compression failure.
pub fn encode_base64_jpeg(raw: &[u8]) -> Result<String, ImageCompressError> {
    Ok(base64::engine::general_purpose::STANDARD.encode(compress_to_jpeg(raw, SIZE_CEILING)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Frame, Rgb, Rgba};

    fn make_png(w: u32, h: u32) -> Vec<u8> {
        let img = RgbImage::from_pixel(w, h, Rgb([100, 150, 200]));
        let mut buf = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    fn make_png_rgba(w: u32, h: u32) -> Vec<u8> {
        let img = RgbaImage::from_pixel(w, h, Rgba([100, 150, 200, 128]));
        let mut buf = Cursor::new(Vec::new());
        DynamicImage::ImageRgba8(img)
            .write_to(&mut buf, ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    fn make_gif(n_frames: u32, w: u32, h: u32) -> Vec<u8> {
        use image::codecs::gif::GifEncoder;
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut encoder = GifEncoder::new(&mut buf);
            for i in 0..n_frames {
                let shade = u8::try_from((i * 40) % 256).unwrap();
                let frame_img = RgbaImage::from_pixel(w, h, Rgba([shade, 0, 0, 255]));
                encoder.encode_frame(Frame::new(frame_img)).unwrap();
            }
        }
        buf
    }

    #[test]
    fn compress_shrinks_oversized() {
        let out = compress_to_jpeg(&make_png(4000, 3000), SIZE_CEILING).unwrap();
        let img = image::load_from_memory(&out).unwrap();
        assert_eq!(img.width().max(img.height()), 1024);
        assert!(out.len() <= SIZE_CEILING);
    }

    #[test]
    fn compress_handles_rgba() {
        let out = compress_to_jpeg(&make_png_rgba(200, 200), SIZE_CEILING).unwrap();
        assert!(!out.is_empty());
    }

    #[test]
    fn compress_small_image_not_upscaled() {
        let out = compress_to_jpeg(&make_png(64, 64), SIZE_CEILING).unwrap();
        let img = image::load_from_memory(&out).unwrap();
        assert!(img.width().max(img.height()) <= 1024);
        assert_eq!((img.width(), img.height()), (64, 64));
    }

    #[test]
    fn compress_raises_when_unreachable() {
        let err = compress_to_jpeg(&make_png(100, 100), 1).unwrap_err();
        assert!(matches!(err, ImageCompressError::TooLarge(1)));
    }

    #[test]
    fn gif_single_frame_no_tiling() {
        let img = open_as_rgb(&make_gif(1, 80, 60)).unwrap();
        assert_eq!((img.width(), img.height()), (80, 60));
    }

    #[test]
    fn gif_animated_tiles_frames_horizontally() {
        let img = open_as_rgb(&make_gif(6, 80, 60)).unwrap();
        assert_eq!(img.height(), 60);
        assert_eq!(img.width(), 80 * u32::try_from(MAX_GIF_FRAMES).unwrap());
    }

    #[test]
    fn gif_fewer_frames_than_max() {
        let img = open_as_rgb(&make_gif(2, 80, 60)).unwrap();
        assert_eq!((img.width(), img.height()), (160, 60));
    }

    #[test]
    fn gif_compress_produces_valid_jpeg() {
        let out = compress_to_jpeg(&make_gif(6, 80, 60), SIZE_CEILING).unwrap();
        assert_eq!(&out[..3], &[0xFF, 0xD8, 0xFF]);
        assert!(out.len() <= SIZE_CEILING);
    }

    #[test]
    fn gif_compress_for_description_produces_valid_jpeg() {
        let out = compress_for_description(&make_gif(6, 80, 60)).unwrap();
        assert_eq!(&out[..3], &[0xFF, 0xD8, 0xFF]);
        assert!(out.len() <= SIZE_CEILING_DESCRIBE);
    }

    #[test]
    fn encode_base64_jpeg_returns_ascii_string() {
        let encoded = encode_base64_jpeg(&make_png(50, 50)).unwrap();
        assert!(encoded.is_ascii());
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
        assert_eq!(&decoded[..3], &[0xFF, 0xD8, 0xFF]);
    }
}
