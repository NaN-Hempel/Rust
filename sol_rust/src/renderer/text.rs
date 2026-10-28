// Draws text with the 8x8 bitmap font from the font8x8 crate.

use font8x8::{UnicodeFonts, BASIC_FONTS};

use crate::renderer::draw::Canvas;

const GLYPH_SIZE: i64 = 8;

// Button symbols the font doesn't have, in its format: one byte per row, top first, and bit 0 is
// the leftmost pixel. Like the letters, they fill the top 7 rows.
const EXTRA_GLYPHS: &[(char, [u8; 8])] = &[
    ('⏪', [0x88, 0xCC, 0xEE, 0xFF, 0xEE, 0xCC, 0x88, 0x00]), // rewind: two triangles pointing left
    ('⏩', [0x11, 0x33, 0x77, 0xFF, 0x77, 0x33, 0x11, 0x00]), // fast-forward: two pointing right
    ('▶', [0x04, 0x0C, 0x1C, 0x3C, 0x1C, 0x0C, 0x04, 0x00]), // play
    ('‖', [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x00]), // pause: two bars
];

pub fn width(text: &str, scale: i64) -> i64 {
    text.chars().count() as i64 * GLYPH_SIZE * scale
}

pub fn height(scale: i64) -> i64 {
    GLYPH_SIZE * scale
}

// Draws text with its top-left corner at (x, y). Each font pixel becomes a scale x scale square.
// Characters the font doesn't have are drawn as '?'.
pub fn draw(canvas: &mut Canvas, x: i64, y: i64, text: &str, colour: u32, scale: i64) {
    for (i, ch) in text.chars().enumerate() {
        let glyph = EXTRA_GLYPHS
            .iter()
            .find(|&&(c, _)| c == ch)
            .map(|&(_, glyph)| glyph)
            .or_else(|| BASIC_FONTS.get(ch))
            .or_else(|| BASIC_FONTS.get('?'))
            .unwrap();
        let left = x + i as i64 * GLYPH_SIZE * scale;
        // Each byte is one row, top first; bit 0 is the leftmost pixel.
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..GLYPH_SIZE {
                if bits >> col & 1 == 1 {
                    canvas.fill_rect(left + col * scale, y + row as i64 * scale, scale, scale, colour);
                }
            }
        }
    }
}
