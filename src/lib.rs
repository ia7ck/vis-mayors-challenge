mod sample_case;

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};
use svg::node::{self, element};
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};

// https://rustwasm.github.io/docs/book/game-of-life/debugging.html#add-logging-to-our-game-of-life
#[allow(unused_macros)]
macro_rules! log {
    ( $( $t:tt )* ) => {
        web_sys::console::log_1(&format!( $( $t )* ).into());
    }
}

// https://atcoder.jp/contests/tessoku-book/tasks/tessoku_book_fr

const N: usize = 50;
const K: usize = 400;
const L: usize = 20;

const W: usize = 800;
const H: usize = 800;
#[rustfmt::skip]
const PALETTE: [&str; L + 1] = [
    "white",
    "#ffffa9",
    "#ba9fe8",
    "#adcf71",
    "#f1c8ff",
    "#69bc91",
    "#ffa6c8",
    "#00d2cf",
    "#ffa292",
    "#2abfad",
    "#ffc496",
    "#82fff5",
    "#d29ea7",
    "#cfffc8",
    "#ffcce5",
    "#aaaf83",
    "#b7e3ff",
    "#baa993",
    "#ccffff",
    "#7cb4c4",
    "#fff9da",
];

#[derive(Serialize, Deserialize)]
struct Data {
    score: u64,
    svg: String,
}

#[wasm_bindgen]
pub fn judge(input: &str, output: &str, colorful: bool) -> Result<JsValue, JsValue> {
    let input = parse_input(input)?;
    let output = parse_output(output)?;

    let score = calc_score(&input, &output);
    let svg = make_svg(&input, &output, colorful);

    let data = Data { score, svg };
    let data = serde_wasm_bindgen::to_value(&data)?;
    Ok(data)
}

#[wasm_bindgen]
pub fn sample_1_input() -> JsValue {
    JsValue::from_str(sample_case::SAMPLE_1_INPUT)
}

#[wasm_bindgen]
pub fn sample_1_output() -> JsValue {
    JsValue::from_str(sample_case::SAMPLE_1_OUTPUT)
}

#[derive(Debug)]
struct Input {
    // ab[0] は使わない
    ab: [(u64, u64); K + 1],
    c: [[usize; N]; N],
}

#[derive(Debug)]
struct Output {
    // area[k] := 地区 k が属する特別区の番号
    // area[0] は使わない
    area: [usize; K + 1],
}

fn parse_input(input: &str) -> Result<Input, String> {
    // N K L
    // A[1] B[1]
    // .
    // .
    // A[K] B[K]
    // C[1][1] C[1][2] ... C[1][N]
    // .
    // .
    // C[N][1] C[N][2] ... C[N][N]

    let input = input.trim();
    let lines = input.lines().collect::<Vec<_>>();
    let n = lines[0].split(' ').nth(0);
    let k = lines[0].split(' ').nth(1);
    let l = lines[0].split(' ').nth(2);
    if n != Some(&format!("{}", N)) || k != Some(&format!("{}", K)) || l != Some(&format!("{}", L))
    {
        return Err("invalid input: Line 1".to_string());
    }

    macro_rules! err_format {
        ($line_number: expr) => {
            format!("invalid input: Line {}", $line_number)
        };
        ($line_number: expr, $message: expr) => {
            format!("invalid input: Line {}: {}", $line_number, $message)
        };
    }

    let mut ab = [(0, 0); K + 1];
    for i in 1..=K {
        let line = lines.get(i).ok_or_else(|| err_format!(i))?;

        let a = line.split(' ').nth(0).ok_or_else(|| err_format!(i))?;
        let b = line.split(' ').nth(1).ok_or_else(|| err_format!(i))?;

        let a = a.parse::<u64>().map_err(|err| err_format!(i, err))?;
        let b = b.parse::<u64>().map_err(|err| err_format!(i, err))?;

        if !(50000 <= a && a <= 100000 && 1000 <= b && b <= 2000) {
            return Err(err_format!(i));
        }

        ab[i] = (a, b);
    }

    let mut c = [[0; N]; N];
    for i in (K + 1)..=(K + N) {
        let line = lines
            .get(i)
            .map(|l| l.split(' ').collect::<Vec<_>>())
            .ok_or_else(|| err_format!(i))?;

        for j in 0..N {
            let x = line.get(j).ok_or_else(|| err_format!(i))?;
            let x = x.parse::<usize>().map_err(|err| err_format!(i, err))?;
            if x > K {
                return Err(err_format!(i));
            }
            c[i - K - 1][j] = x;
        }
    }
    // TODO: 行数・列数が多い場合?

    let input = Input { ab, c };
    check_input(&input)?;

    Ok(input)
}

fn check_input(input: &Input) -> Result<(), String> {
    // KYOPRO 市全体 (c[i][j] != 0 の部分) は連結

    let mut total_area = 0;
    for i in 0..N {
        for j in 0..N {
            if input.c[i][j] >= 1 {
                total_area += 1;
            }
        }
    }

    let mut visited = [[false; N]; N];
    let mut que = VecDeque::new();
    'l: for i in 0..N {
        for j in 0..N {
            if input.c[i][j] >= 1 {
                visited[i][j] = true;
                que.push_back((i, j));
                break 'l;
            }
        }
    }
    while let Some((i, j)) = que.pop_front() {
        macro_rules! push {
            ($ni: expr, $nj: expr) => {
                if !visited[$ni][$nj] && input.c[$ni][$nj] >= 1 {
                    visited[$ni][$nj] = true;
                    que.push_back(($ni, $nj));
                }
            };
        }
        if i >= 1 {
            push!(i - 1, j);
        }
        if j >= 1 {
            push!(i, j - 1);
        }
        if i + 1 < N {
            push!(i + 1, j);
        }
        if j + 1 < N {
            push!(i, j + 1);
        }
    }
    let mut visited_area = 0;
    for i in 0..N {
        for j in 0..N {
            if visited[i][j] {
                visited_area += 1;
            }
        }
    }
    assert!(visited_area <= total_area);
    if visited_area != total_area {
        return Err("constraints violation: KYOPRO City disconnected".to_string());
    }

    // 地区 1, 2, ..., K はすべて連結

    let mut total_area = [0; K + 1];
    for i in 0..N {
        for j in 0..N {
            if input.c[i][j] >= 1 {
                total_area[input.c[i][j]] += 1;
            }
        }
    }

    let mut visited = [[false; N]; N];
    let mut visited_area = [0; K + 1];
    for i in 0..N {
        for j in 0..N {
            let c_ij = input.c[i][j];
            if c_ij == 0 {
                continue;
            }
            if visited[i][j] {
                continue;
            }
            if visited_area[c_ij] >= 1 {
                assert_eq!(visited_area[c_ij], total_area[c_ij]);
                continue;
            }
            let mut que = VecDeque::new();
            visited[i][j] = true;
            visited_area[c_ij] += 1;
            que.push_back((i, j));
            while let Some((y, x)) = que.pop_front() {
                macro_rules! push {
                    ($ny: expr, $nx: expr) => {
                        if !visited[$ny][$nx] && input.c[$ny][$nx] == c_ij {
                            visited[$ny][$nx] = true;
                            visited_area[input.c[$ny][$nx]] += 1;
                            que.push_back(($ny, $nx));
                        }
                    };
                }
                if y >= 1 {
                    push!(y - 1, x);
                }
                if x >= 1 {
                    push!(y, x - 1);
                }
                if y + 1 < N {
                    push!(y + 1, x);
                }
                if x + 1 < N {
                    push!(y, x + 1);
                }
            }
            assert!(visited_area[c_ij] <= total_area[c_ij]);
            if visited_area[c_ij] != total_area[c_ij] {
                return Err(format!("constraints violation: Area {} disconnected", c_ij));
            }
        }
    }

    Ok(())
}

fn parse_output(output: &str) -> Result<Output, String> {
    let mut area = [0; K + 1];
    let mut special_area = [false; L + 1];

    let lines = output.trim().lines().collect::<Vec<_>>();
    for i in 0..K {
        let line = lines
            .get(i)
            .ok_or_else(|| format!("invalid output: Line {}", i + 1))?;
        let y = line
            .parse::<usize>()
            .map_err(|err| format!("invalid output: Line {}: {}", i + 1, err))?;
        if y > L {
            return Err(format!("invalid output: Line {}", i + 1));
        }
        area[i + 1] = y;
        special_area[y] = true;
    }

    // check output

    for i in 1..=L {
        if !special_area[i] {
            return Err(format!("invalid output: Special Area {} is empty", i));
        }
    }

    Ok(Output { area })
}

fn calc_score(input: &Input, output: &Output) -> u64 {
    let special_area_component_size = calc_special_area_component_size(input, output);
    let each_special_area_connected = special_area_component_size[1..=L]
        .iter()
        .all(|&size| size == 1);
    let base = if each_special_area_connected {
        1e6
    } else {
        1e3
    };

    let mut p = [0; L + 1];
    let mut q = [0; L + 1];
    for i in 1..=K {
        let (a, b) = input.ab[i];
        p[output.area[i]] += a;
        q[output.area[i]] += b;
    }
    let p_min = p[1..=L].iter().min().copied().unwrap();
    let p_max = p[1..=L].iter().max().copied().unwrap();
    let q_min = q[1..=L].iter().min().copied().unwrap();
    let q_max = q[1..=L].iter().max().copied().unwrap();

    (base * (p_min as f64 / p_max as f64).min(q_min as f64 / q_max as f64)).round() as u64
}

fn calc_special_area_component_size(input: &Input, output: &Output) -> [usize; L + 1] {
    let mut visited = [[false; N]; N];
    let mut size = [0; L + 1];
    for i in 0..N {
        for j in 0..N {
            if input.c[i][j] == 0 {
                continue;
            }
            if visited[i][j] {
                continue;
            }
            visited[i][j] = true;
            let d = output.area[input.c[i][j]];
            size[d] += 1;
            let mut que = VecDeque::new();
            que.push_back((i, j));
            while let Some((y, x)) = que.pop_front() {
                macro_rules! push {
                    ($ny: expr, $nx: expr) => {
                        if !visited[$ny][$nx] && output.area[input.c[$ny][$nx]] == d {
                            visited[$ny][$nx] = true;
                            que.push_back(($ny, $nx));
                        }
                    };
                }
                if y >= 1 {
                    push!(y - 1, x);
                }
                if x >= 1 {
                    push!(y, x - 1);
                }
                if y + 1 < N {
                    push!(y + 1, x);
                }
                if x + 1 < N {
                    push!(y, x + 1);
                }
            }
        }
    }
    size
}

fn make_svg(input: &Input, output: &Output, colorful: bool) -> String {
    let line = |x1: usize, y1: usize, x2: usize, y2: usize| {
        element::Line::new()
            .set("x1", x1)
            .set("y1", y1)
            .set("x2", x2)
            .set("y2", y2)
            .set("stroke", "black")
    };

    let mut area_border = Vec::new();
    let mut visited = [[false; N]; N];
    for i in 0..N {
        for j in 0..N {
            let c_ij = input.c[i][j];
            if c_ij == 0 {
                continue;
            }
            if visited[i][j] {
                continue;
            }
            let mut que = VecDeque::new();
            visited[i][j] = true;
            que.push_back((i, j));
            while let Some((y, x)) = que.pop_front() {
                macro_rules! push {
                    ($ny: expr, $nx: expr) => {
                        if !visited[$ny][$nx] && input.c[$ny][$nx] == c_ij {
                            visited[$ny][$nx] = true;
                            que.push_back(($ny, $nx));
                        }
                    };
                }
                if y >= 1 {
                    push!(y - 1, x);
                }
                if x >= 1 {
                    push!(y, x - 1);
                }
                if y + 1 < N {
                    push!(y + 1, x);
                }
                if x + 1 < N {
                    push!(y, x + 1);
                }
                if y == 0 || input.c[y - 1][x] != c_ij {
                    area_border.push(
                        line(W / N * x, H / N * y, W / N * (x + 1), H / N * y)
                            .set("stroke-dasharray", "2, 4, 4, 4, 2")
                            .set("stroke-linecap", "round"),
                    );
                }
                if x == 0 || input.c[y][x - 1] != c_ij {
                    area_border.push(
                        line(W / N * x, H / N * y, W / N * x, H / N * (y + 1))
                            .set("stroke-dasharray", "2, 4, 4, 4, 2")
                            .set("stroke-linecap", "round"),
                    );
                }
                if y + 1 == N || input.c[y + 1][x] != c_ij {
                    area_border.push(
                        line(W / N * x, H / N * (y + 1), W / N * (x + 1), H / N * (y + 1))
                            .set("stroke-dasharray", "2, 4, 4, 4, 2")
                            .set("stroke-linecap", "round"),
                    );
                }
                if x + 1 == N || input.c[y][x + 1] != c_ij {
                    area_border.push(
                        line(W / N * (x + 1), H / N * y, W / N * (x + 1), H / N * (y + 1))
                            .set("stroke-dasharray", "2, 4, 4, 4, 2")
                            .set("stroke-linecap", "round"),
                    );
                }
            }
        }
    }

    let mut d = [[0; N]; N];
    for i in 0..N {
        for j in 0..N {
            if input.c[i][j] == 0 {
                continue;
            }
            d[i][j] = output.area[input.c[i][j]];
        }
    }

    let mut special_area_border = Vec::new();
    let mut visited = [[false; N]; N];
    for i in 0..N {
        for j in 0..N {
            if d[i][j] == 0 {
                continue;
            }
            if visited[i][j] {
                continue;
            }
            visited[i][j] = true;
            let mut que = VecDeque::new();
            que.push_back((i, j));
            while let Some((y, x)) = que.pop_front() {
                macro_rules! push {
                    ($ny: expr, $nx: expr) => {
                        if !visited[$ny][$nx] && d[$ny][$nx] == d[i][j] {
                            visited[$ny][$nx] = true;
                            que.push_back(($ny, $nx));
                        }
                    };
                }
                if y >= 1 {
                    push!(y - 1, x);
                }
                if x >= 1 {
                    push!(y, x - 1);
                }
                if y + 1 < N {
                    push!(y + 1, x);
                }
                if x + 1 < N {
                    push!(y, x + 1);
                }
                if y == 0 || d[y - 1][x] != d[i][j] {
                    special_area_border.push(
                        line(W / N * x, H / N * y, W / N * (x + 1), H / N * y)
                            .set("stroke-width", 2)
                            .set("stroke-linecap", "round"),
                    );
                }
                if x == 0 || d[y][x - 1] != d[i][j] {
                    special_area_border.push(
                        line(W / N * x, H / N * y, W / N * x, H / N * (y + 1))
                            .set("stroke-width", 2)
                            .set("stroke-linecap", "round"),
                    );
                }
                if y + 1 == N || d[y + 1][x] != d[i][j] {
                    special_area_border.push(
                        line(W / N * x, H / N * (y + 1), W / N * (x + 1), H / N * (y + 1))
                            .set("stroke-width", 2)
                            .set("stroke-linecap", "round"),
                    );
                }
                if x + 1 == N || d[y][x + 1] != d[i][j] {
                    special_area_border.push(
                        line(W / N * (x + 1), H / N * y, W / N * (x + 1), H / N * (y + 1))
                            .set("stroke-width", 2)
                            .set("stroke-linecap", "round"),
                    );
                }
            }
        }
    }

    let special_area_component_size = calc_special_area_component_size(input, output);

    let mut doc = svg::Document::new()
        .set("id", "vis-svg")
        .set("viewBox", (-2, -2, W + 4, H + 4))
        .set("width", W)
        .set("height", H);

    // N × N のグリッド
    for i in 0..N {
        for j in 0..N {
            let c = input.c[i][j];
            let (a, b) = input.ab[c];
            let text = if c == 0 {
                format!("({}, {})", i + 1, j + 1)
            } else {
                format!(
                    "({}, {})\na={}\nb={}\narea {}\nSP area {}",
                    i + 1,
                    j + 1,
                    a,
                    b,
                    c,
                    d[i][j]
                )
            };
            let title = node::Text::new(text);
            let title = element::Title::new().add(title);
            let fill = if colorful {
                PALETTE[d[i][j]]
            } else if special_area_component_size[d[i][j]] == 1 {
                "gainsboro"
            } else {
                "white"
            };
            let rect = element::Rectangle::new()
                .set("x", W / N * j)
                .set("y", H / N * i)
                .set("width", W / N)
                .set("height", H / N)
                .set("fill", fill)
                .set("stroke", "black")
                .set("stroke-opacity", 0.05);
            let group = element::Group::new().add(title).add(rect);
            doc = doc.add(group);
        }
    }

    for line in area_border {
        doc = doc.add(line);
    }
    for line in special_area_border {
        doc = doc.add(line);
    }

    doc.to_string()
}

#[cfg(test)]
mod tests {
    use super::{calc_score, parse_input, parse_output, sample_case};

    #[test]
    fn sample_1_test() {
        assert!(parse_input(sample_case::SAMPLE_1_INPUT).is_ok());
        assert!(parse_output(sample_case::SAMPLE_1_OUTPUT).is_ok());
        assert_eq!(
            calc_score(
                &parse_input(sample_case::SAMPLE_1_INPUT).unwrap(),
                &parse_output(sample_case::SAMPLE_1_OUTPUT).unwrap()
            ),
            432650
        );
    }
}
