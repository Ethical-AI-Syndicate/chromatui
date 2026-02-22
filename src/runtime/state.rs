use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeState {
    Normal,
    Inline(InlineContext),
    Suspended,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlineContext {
    pub prompt: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timer {
    pub id: u64,
    pub interval_ms: u64,
}

pub struct Runtime {
    state: RuntimeState,
    timers: HashMap<u64, Timer>,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            state: RuntimeState::Normal,
            timers: HashMap::new(),
        }
    }

    pub fn state(&self) -> &RuntimeState {
        &self.state
    }

    pub fn add_timer(&mut self, interval_ms: u64, id: u64) {
        self.timers.insert(id, Timer { id, interval_ms });
    }

    pub fn has_timer(&self, id: u64) -> bool {
        self.timers.contains_key(&id)
    }

    pub fn remove_timer(&mut self, id: u64) {
        self.timers.remove(&id);
    }

    pub fn push_inline_mode(&mut self, prompt: &str) {
        self.state = RuntimeState::Inline(InlineContext {
            prompt: prompt.to_string(),
        });
    }

    pub fn pop_inline_mode(&mut self) -> String {
        let old = std::mem::replace(&mut self.state, RuntimeState::Normal);
        if let RuntimeState::Inline(ctx) = old {
            ctx.prompt
        } else {
            String::new()
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}
