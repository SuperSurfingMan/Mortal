use super::{Agent, BatchifiedAgent, InvisibleState};
use crate::arena::GameResult;
use crate::mjai::{Event, EventExt, EventWithCanAct, Metadata};
use crate::state::PlayerState;
use std::env;
use std::ffi::{OsStr, OsString};
use std::io::prelude::*;
use std::io::{BufReader, Lines};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use serde_json as json;

pub struct AkochanAgent {
    player_id: u8,
    child: Child,
    stdin: ChildStdin,
    stdout_lines: Lines<BufReader<ChildStdout>>,

    event_idx: usize,
    naki_tx: Option<Event>,
}

impl AkochanAgent {
    pub fn new(player_id: u8) -> Result<Self> {
        ensure!(matches!(player_id, 0..=3));

        let akochan_dir = env::var_os("AKOCHAN_DIR").unwrap_or_else(|| OsString::from("akochan"));
        let akochan_exe = [&akochan_dir, OsStr::new("system.exe")]
            .iter()
            .collect::<PathBuf>();
        let akochan_tactics =
            env::var_os("AKOCHAN_TACTICS").unwrap_or_else(|| OsString::from("tactics.json"));

        let mut child = Command::new(akochan_exe)
            .arg("pipe")
            .arg(akochan_tactics)
            .arg(player_id.to_string())
            .current_dir(akochan_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("failed to spawn akochan")?;

        let stdin = child
            .stdin
            .take()
            .context("failed to get stdin of akochan")?;
        let stdout = child
            .stdout
            .take()
            .context("failed to get stdout of akochan")?;
        let stdout_lines = BufReader::new(stdout).lines();

        Ok(Self {
            player_id,
            child,
            stdin,
            stdout_lines,

            event_idx: 0,
            naki_tx: None,
        })
    }

    pub fn new_batched(player_ids: &[u8]) -> Result<BatchifiedAgent<Self>> {
        BatchifiedAgent::new(Self::new, player_ids)
    }
}

impl Drop for AkochanAgent {
    fn drop(&mut self) {
        if let Err(err) = self.child.kill() {
            log::error!("failed to kill akochan: {err}");
        }
        if let Err(err) = self.child.wait() {
            log::error!("failed to wait akochan: {err}");
        }
    }
}

impl Agent for AkochanAgent {
    fn name(&self) -> String {
        "akochan".to_owned()
    }

    fn react(
        &mut self,
        events: &[EventExt],
        state: &PlayerState,
        _: Option<InvisibleState>,
    ) -> Result<EventExt> {
        // handle two-phase actions like Chi, Pon and Riichi
        if let Some(dahai) = self.naki_tx.take() {
            let last = events.last().context("events is empty")?;
            match last.event {
                Event::Chi { actor, .. }
                | Event::Pon { actor, .. }
                | Event::Daiminkan { actor, .. }
                | Event::Reach { actor, .. }
                    if actor == self.player_id =>
                {
                    return Ok(EventExt::no_meta(dahai));
                }
                _ => (),
            };
        }

        let start = Instant::now();
        let sent_new = self.event_idx < events.len();
        for i in self.event_idx..events.len() {
            let v = EventWithCanAct {
                event: events[i].event.clone(),
                can_act: Some(i == events.len() - 1),
            };
            json::to_writer(&mut self.stdin, &v)?;
            self.stdin.write_all(b"\n")?;
            self.stdin.flush()?;
        }
        self.event_idx = events.len();

        // akochan 只在「自己的 tsumo」或「别人 dahai/kakan」时回话；其余批次它 不会输出任何行。
        // 但本实现每写完一批就必读一行，且 arena 会同一时刻把 3 个 akochan 一起问（其余两个
        // 的最后事件不是触发点）→ 永不回话 → 整局死锁。这里先判定触发点：非触发点直接
        // 返回 Event::None（相当于 pass），**不读子进程**，以免请求/应答错位。
        // 注意：`set_scene` 可能用同一份 log 重复调用（没有新事件）——此时 akochan 什么都不会回，
        //       必须直接返回 None，否则又会阻塞。
        let triggered = sent_new
            && match events.last().map(|e| &e.event) {
                Some(Event::Tsumo { actor, .. }) => *actor == self.player_id,
                Some(Event::Dahai { actor, .. } | Event::Kakan { actor, .. }) => {
                    *actor != self.player_id
                }
                _ => false,
            };
        if !triggered {
            if std::env::var_os("AKO_TRACE").is_some() {
                eprintln!("[ako p{}] n={} idx={} last={:?} -> None",
                    self.player_id, events.len(), self.event_idx, events.last().map(|e| &e.event));
            }
            return Ok(EventExt::no_meta(Event::None));
        }

        let line = self
            .stdout_lines
            .next()
            .context("failed to read from akochan: unexpected EOF")?
            .context("failed to read from akochan")?;
        let mut actions = json::from_str::<Vec<Event>>(&line)
            .context("failed to parse JSON output of akochan")?
            .into_iter();

        let ev = actions.next().context("output is empty")?;
        if let Some(naki_tx) = actions.next() {
            self.naki_tx = Some(naki_tx);
        }

        let eval_time_ns = Instant::now()
            .checked_duration_since(start)
            .unwrap_or(Duration::ZERO)
            .as_nanos()
            .try_into()
            .unwrap_or(u64::MAX);
        Ok(EventExt {
            event: ev,
            meta: Some(Metadata {
                eval_time_ns: Some(eval_time_ns),
                shanten: Some(state.shanten()),
                ..Default::default()
            }),
        })
    }

    fn start_game(&mut self) -> Result<()> {
        let start_game = json::json!({
            "type": "start_game",
            "kyoku_first": 0,
            "aka_flag": true,
        });
        json::to_writer(&mut self.stdin, &start_game)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn end_kyoku(&mut self) -> Result<()> {
        json::to_writer(&mut self.stdin, &Event::EndKyoku)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        self.event_idx = 0;
        self.naki_tx = None;
        Ok(())
    }

    fn end_game(&mut self, _: &GameResult) -> Result<()> {
        json::to_writer(&mut self.stdin, &Event::EndGame)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        Ok(())
    }
}
