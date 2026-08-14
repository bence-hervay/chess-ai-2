//! Game implementations. Each game is a static Rust module implementing
//! [`crate::game::Game`]; there is no plugin system.

pub mod breakthrough;
pub mod chess;
pub mod connect_k;
pub mod othello;

use serde::{Deserialize, Serialize};

/// Typed, fully explicit game selection used in experiment configurations.
///
/// The CLI matches on this enum and dispatches to generic functions; the
/// search/training/evaluation code never sees it.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum GameSpec {
    #[serde(rename = "connect_k")]
    ConnectK {
        width: u16,
        height: u16,
        k: u16,
        gravity: bool,
    },
    #[serde(rename = "breakthrough")]
    Breakthrough { width: u16, height: u16, rows: u16 },
    #[serde(rename = "othello")]
    Othello { width: u16, height: u16 },
    #[serde(rename = "chess")]
    Chess {},
}

impl GameSpec {
    /// Short label used in run-directory names.
    pub fn label(&self) -> String {
        match self {
            GameSpec::ConnectK {
                width,
                height,
                k,
                gravity,
            } => {
                let g = if *gravity { "g" } else { "f" };
                format!("connectk-{width}x{height}k{k}{g}")
            }
            GameSpec::Breakthrough {
                width,
                height,
                rows,
            } => {
                format!("breakthrough-{width}x{height}r{rows}")
            }
            GameSpec::Othello { width, height } => {
                format!("othello-{width}x{height}")
            }
            GameSpec::Chess {} => "chess".to_string(),
        }
    }
}
