//! String-art [`SolidMap`] for the module's own tests.
//!
//! ```text
//! StringMap::new(&["####",
//!                  "#..#",
//!                  "####"])
//! ```
//!
//! Row index is `tz` (row 0 is `tz = 0`, so **Z increases downward** in the
//! literal), column index is `tx`, `'#'` is solid and every other character is
//! free. Anything outside the rectangle is solid, per the [`SolidMap`] contract.

use super::map::SolidMap;

pub(super) struct StringMap {
    rows: Vec<Vec<bool>>,
    width: i32,
    height: i32,
}

impl StringMap {
    pub(super) fn new(rows: &[&str]) -> Self {
        let width = rows.iter().map(|r| r.len()).max().unwrap_or(0) as i32;
        let height = rows.len() as i32;
        let rows = rows
            .iter()
            .map(|r| {
                let mut row: Vec<bool> = r.chars().map(|c| c == '#').collect();
                row.resize(width as usize, true);
                row
            })
            .collect();
        Self {
            rows,
            width,
            height,
        }
    }
}

impl SolidMap for StringMap {
    fn is_solid(&self, tx: i32, tz: i32) -> bool {
        if tx < 0 || tz < 0 || tx >= self.width || tz >= self.height {
            return true; // out of bounds is solid
        }
        self.rows[tz as usize][tx as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rows_and_seals_the_border() {
        let m = StringMap::new(&["####", "#..#", "####"]);
        assert!(m.is_solid(0, 0));
        assert!(!m.is_solid(1, 1));
        assert!(!m.is_solid(2, 1));
        assert!(m.is_solid(3, 1));
        // Out of bounds in every direction.
        assert!(m.is_solid(-1, 1));
        assert!(m.is_solid(4, 1));
        assert!(m.is_solid(1, -1));
        assert!(m.is_solid(1, 3));
    }

    #[test]
    fn short_rows_are_padded_solid() {
        let m = StringMap::new(&["....", ".."]);
        assert!(!m.is_solid(3, 0));
        assert!(m.is_solid(3, 1));
    }
}
