use chromatui_core::Key;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InlineAction {
    Submit(String),
    Cancel(String),
    Complete(String),
}

pub struct InlineEditor {
    prompt: String,
    content: String,
    cursor_pos: usize,
    history: Vec<String>,
    history_index: Option<usize>,
}

impl InlineEditor {
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            content: String::new(),
            cursor_pos: 0,
            history: Vec::new(),
            history_index: None,
        }
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn cursor_pos(&self) -> usize {
        self.cursor_pos
    }

    pub fn handle_key(&mut self, key: Key) -> Option<InlineAction> {
        match key {
            Key::Char(c) => {
                if self.cursor_pos > self.content.len() {
                    self.cursor_pos = self.content.len();
                }
                self.content.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
                self.history_index = None;
                None
            }
            Key::Backspace => {
                if self.cursor_pos > 0 && !self.content.is_empty() {
                    self.content.remove(self.cursor_pos - 1);
                    self.cursor_pos = self.cursor_pos.saturating_sub(1);
                }
                self.history_index = None;
                None
            }
            Key::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
                None
            }
            Key::Right => {
                self.cursor_pos = (self.cursor_pos + 1).min(self.content.len());
                None
            }
            Key::Home => {
                self.cursor_pos = 0;
                None
            }
            Key::End => {
                self.cursor_pos = self.content.len();
                None
            }
            Key::Enter => {
                let submitted = self.content.clone();
                if !submitted.is_empty() {
                    self.history.push(submitted.clone());
                }
                let result = Some(InlineAction::Submit(std::mem::take(&mut self.content)));
                self.cursor_pos = 0;
                self.history_index = None;
                result
            }
            Key::Escape => {
                let cancelled = self.content.clone();
                self.content.clear();
                self.cursor_pos = 0;
                Some(InlineAction::Cancel(cancelled))
            }
            Key::Up => {
                if let Some(idx) = self.history_index {
                    if idx + 1 < self.history.len() {
                        self.history_index = Some(idx + 1);
                    }
                } else if !self.history.is_empty() {
                    self.history_index = Some(0);
                }
                if let Some(idx) = self.history_index {
                    self.content = self.history[idx].clone();
                    self.cursor_pos = self.content.len();
                }
                None
            }
            Key::Down => {
                if let Some(idx) = self.history_index {
                    if idx == 0 {
                        self.history_index = None;
                        self.content.clear();
                    } else {
                        self.history_index = Some(idx - 1);
                        self.content = self.history[idx - 1].clone();
                    }
                }
                self.cursor_pos = self.content.len();
                None
            }
            _ => None,
        }
    }

    pub fn render(&self) -> String {
        let mut output = self.prompt.clone();
        output.push_str(&self.content);
        output
    }

    pub fn submit(&self) -> String {
        self.content.clone()
    }

    pub fn cancel(&self) -> String {
        self.content.clone()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WidgetId(pub usize);

impl WidgetId {
    pub fn new(id: usize) -> Self {
        WidgetId(id)
    }
}

pub trait Focusable {
    fn widget_id(&self) -> WidgetId;
    fn is_focusable(&self) -> bool;
    fn on_focus(&mut self);
    fn on_blur(&mut self);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusDirection {
    Next,
    Previous,
    Up,
    Down,
    Left,
    Right,
}

pub struct FocusManager {
    focused_widget: Option<WidgetId>,
    focus_order: Vec<WidgetId>,
    focus_enabled: bool,
}

impl FocusManager {
    pub fn new() -> Self {
        Self {
            focused_widget: None,
            focus_order: Vec::new(),
            focus_enabled: true,
        }
    }

    pub fn register(&mut self, widget_id: WidgetId) {
        if !self.focus_order.contains(&widget_id) {
            self.focus_order.push(widget_id);
        }
    }

    pub fn unregister(&mut self, widget_id: WidgetId) {
        self.focus_order.retain(|id| *id != widget_id);
        if self.focused_widget == Some(widget_id) {
            self.focused_widget = None;
        }
    }

    pub fn focus(&mut self, widget_id: WidgetId) {
        if self.focus_order.contains(&widget_id) {
            self.focused_widget = Some(widget_id);
        }
    }

    pub fn blur(&mut self) {
        self.focused_widget = None;
    }

    pub fn focused(&self) -> Option<WidgetId> {
        self.focused_widget
    }

    pub fn move_focus(&mut self, direction: FocusDirection) -> Option<WidgetId> {
        if !self.focus_enabled || self.focus_order.is_empty() {
            return None;
        }

        let current_idx = self
            .focused_widget
            .and_then(|id| self.focus_order.iter().position(|&x| x == id));

        let new_idx = match (current_idx, direction) {
            (None, _) => Some(0),
            (Some(idx), FocusDirection::Next | FocusDirection::Down | FocusDirection::Right) => {
                if idx + 1 < self.focus_order.len() {
                    Some(idx + 1)
                } else {
                    Some(0)
                }
            }
            (Some(idx), FocusDirection::Previous | FocusDirection::Up | FocusDirection::Left) => {
                if idx > 0 {
                    Some(idx - 1)
                } else {
                    Some(self.focus_order.len() - 1)
                }
            }
        };

        new_idx.map(|idx| {
            let id = self.focus_order[idx];
            self.focused_widget = Some(id);
            id
        })
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.focus_enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.focus_enabled
    }
}

impl Default for FocusManager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ButtonState {
    Normal,
    Hovered,
    Pressed,
    Disabled,
}

pub struct Button {
    pub id: WidgetId,
    pub label: String,
    pub state: ButtonState,
    pub shortcut: Option<Key>,
}

impl Button {
    pub fn new(id: WidgetId, label: &str) -> Self {
        Self {
            id,
            label: label.to_string(),
            state: ButtonState::Normal,
            shortcut: None,
        }
    }

    pub fn with_shortcut(mut self, key: Key) -> Self {
        self.shortcut = Some(key);
        self
    }

    pub fn pressed(&mut self) -> bool {
        if self.state == ButtonState::Pressed {
            self.state = ButtonState::Normal;
            true
        } else {
            false
        }
    }

    pub fn is_disabled(&self) -> bool {
        self.state == ButtonState::Disabled
    }
}

impl Focusable for Button {
    fn widget_id(&self) -> WidgetId {
        self.id
    }

    fn is_focusable(&self) -> bool {
        self.state != ButtonState::Disabled
    }

    fn on_focus(&mut self) {
        if self.state != ButtonState::Disabled {
            self.state = ButtonState::Hovered;
        }
    }

    fn on_blur(&mut self) {
        if self.state != ButtonState::Disabled {
            self.state = ButtonState::Normal;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputState {
    Normal,
    Focused,
    Disabled,
}

pub struct Input {
    pub id: WidgetId,
    pub value: String,
    pub placeholder: String,
    pub state: InputState,
    pub cursor_pos: usize,
    pub max_length: Option<usize>,
}

impl Input {
    pub fn new(id: WidgetId) -> Self {
        Self {
            id,
            value: String::new(),
            placeholder: String::new(),
            state: InputState::Normal,
            cursor_pos: 0,
            max_length: None,
        }
    }

    pub fn with_placeholder(mut self, placeholder: &str) -> Self {
        self.placeholder = placeholder.to_string();
        self
    }

    pub fn with_max_length(mut self, max: usize) -> Self {
        self.max_length = Some(max);
        self
    }

    pub fn handle_key(&mut self, key: Key) -> bool {
        match key {
            Key::Char(c) => {
                if let Some(max) = self.max_length {
                    if self.value.len() >= max {
                        return false;
                    }
                }
                self.value.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
                true
            }
            Key::Backspace => {
                if self.cursor_pos > 0 && !self.value.is_empty() {
                    self.value.remove(self.cursor_pos - 1);
                    self.cursor_pos = self.cursor_pos.saturating_sub(1);
                }
                true
            }
            Key::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
                true
            }
            Key::Right => {
                self.cursor_pos = (self.cursor_pos + 1).min(self.value.len());
                true
            }
            Key::Home => {
                self.cursor_pos = 0;
                true
            }
            Key::End => {
                self.cursor_pos = self.value.len();
                true
            }
            Key::Delete => {
                if self.cursor_pos < self.value.len() {
                    self.value.remove(self.cursor_pos);
                }
                true
            }
            _ => false,
        }
    }

    pub fn clear(&mut self) {
        self.value.clear();
        self.cursor_pos = 0;
    }
}

impl Focusable for Input {
    fn widget_id(&self) -> WidgetId {
        self.id
    }

    fn is_focusable(&self) -> bool {
        self.state != InputState::Disabled
    }

    fn on_focus(&mut self) {
        if self.state != InputState::Disabled {
            self.state = InputState::Focused;
            self.cursor_pos = self.value.len();
        }
    }

    fn on_blur(&mut self) {
        if self.state != InputState::Disabled {
            self.state = InputState::Normal;
        }
    }
}

pub struct ListItem {
    pub id: WidgetId,
    pub label: String,
    pub selected: bool,
    pub expanded: bool,
}

impl ListItem {
    pub fn new(id: WidgetId, label: &str) -> Self {
        Self {
            id,
            label: label.to_string(),
            selected: false,
            expanded: false,
        }
    }

    pub fn select(&mut self) {
        self.selected = true;
    }

    pub fn deselect(&mut self) {
        self.selected = false;
    }

    pub fn toggle(&mut self) {
        self.selected = !self.selected;
    }
}

pub struct List {
    pub id: WidgetId,
    pub items: Vec<ListItem>,
    pub focused_index: usize,
    pub multiple: bool,
}

impl List {
    pub fn new(id: WidgetId) -> Self {
        Self {
            id,
            items: Vec::new(),
            focused_index: 0,
            multiple: false,
        }
    }

    pub fn with_items(mut self, labels: Vec<&str>) -> Self {
        for (i, label) in labels.into_iter().enumerate() {
            self.items.push(ListItem::new(WidgetId::new(i), label));
        }
        self
    }

    pub fn multiple(mut self) -> Self {
        self.multiple = true;
        self
    }

    pub fn add_item(&mut self, label: &str) -> WidgetId {
        let id = WidgetId::new(self.items.len());
        self.items.push(ListItem::new(id, label));
        id
    }

    pub fn select(&mut self, index: usize) {
        if index >= self.items.len() {
            return;
        }

        if self.multiple {
            self.items[index].toggle();
        } else {
            for item in &mut self.items {
                item.deselect();
            }
            self.items[index].select();
        }
    }

    pub fn selected_items(&self) -> Vec<&ListItem> {
        self.items.iter().filter(|i| i.selected).collect()
    }

    pub fn move_selection(&mut self, direction: FocusDirection) {
        let new_idx = match direction {
            FocusDirection::Up | FocusDirection::Previous => {
                if self.focused_index > 0 {
                    Some(self.focused_index - 1)
                } else {
                    Some(self.items.len() - 1)
                }
            }
            FocusDirection::Down | FocusDirection::Next => {
                if self.focused_index + 1 < self.items.len() {
                    Some(self.focused_index + 1)
                } else {
                    Some(0)
                }
            }
            _ => None,
        };

        if let Some(idx) = new_idx {
            self.focused_index = idx;
        }
    }
}

impl Focusable for List {
    fn widget_id(&self) -> WidgetId {
        self.id
    }

    fn is_focusable(&self) -> bool {
        !self.items.is_empty()
    }

    fn on_focus(&mut self) {}
    fn on_blur(&mut self) {}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckboxState {
    Unchecked,
    Checked,
    Indeterminate,
}

pub struct Checkbox {
    pub id: WidgetId,
    pub label: String,
    pub state: CheckboxState,
}

impl Checkbox {
    pub fn new(id: WidgetId, label: &str) -> Self {
        Self {
            id,
            label: label.to_string(),
            state: CheckboxState::Unchecked,
        }
    }

    pub fn checked(&self) -> bool {
        self.state == CheckboxState::Checked
    }

    pub fn toggle(&mut self) {
        self.state = match self.state {
            CheckboxState::Unchecked => CheckboxState::Checked,
            CheckboxState::Checked => CheckboxState::Unchecked,
            CheckboxState::Indeterminate => CheckboxState::Checked,
        };
    }

    pub fn set_checked(&mut self, checked: bool) {
        self.state = if checked {
            CheckboxState::Checked
        } else {
            CheckboxState::Unchecked
        };
    }
}

impl Focusable for Checkbox {
    fn widget_id(&self) -> WidgetId {
        self.id
    }

    fn is_focusable(&self) -> bool {
        true
    }

    fn on_focus(&mut self) {}
    fn on_blur(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_focus_manager_new() {
        let fm = FocusManager::new();
        assert!(fm.focused().is_none());
    }

    #[test]
    fn test_focus_manager_register() {
        let mut fm = FocusManager::new();
        fm.register(WidgetId::new(1));
        fm.register(WidgetId::new(2));
        assert_eq!(fm.focus_order.len(), 2);
    }

    #[test]
    fn test_focus_manager_focus() {
        let mut fm = FocusManager::new();
        fm.register(WidgetId::new(1));
        fm.register(WidgetId::new(2));
        fm.focus(WidgetId::new(1));
        assert_eq!(fm.focused(), Some(WidgetId::new(1)));
    }

    #[test]
    fn test_focus_manager_move_next() {
        let mut fm = FocusManager::new();
        fm.register(WidgetId::new(1));
        fm.register(WidgetId::new(2));
        fm.focus(WidgetId::new(1));
        let new_focus = fm.move_focus(FocusDirection::Next);
        assert_eq!(new_focus, Some(WidgetId::new(2)));
    }

    #[test]
    fn test_button_pressed() {
        let mut button = Button::new(WidgetId::new(0), "Click me");
        button.state = ButtonState::Pressed;
        assert!(button.pressed());
        assert_eq!(button.state, ButtonState::Normal);
    }

    #[test]
    fn test_input_handle_key() {
        let mut input = Input::new(WidgetId::new(0));
        input.handle_key(Key::Char('h'));
        input.handle_key(Key::Char('i'));
        assert_eq!(input.value, "hi");
    }

    #[test]
    fn test_input_cursor_movement() {
        let mut input = Input::new(WidgetId::new(0));
        input.value = "hello".to_string();
        input.cursor_pos = 5;
        input.handle_key(Key::Left);
        assert_eq!(input.cursor_pos, 4);
    }

    #[test]
    fn test_list_item_selection() {
        let mut list = List::new(WidgetId::new(0));
        list.add_item("Item 1");
        list.add_item("Item 2");
        list.select(0);
        assert!(list.items[0].selected);
    }

    #[test]
    fn test_checkbox_toggle() {
        let mut checkbox = Checkbox::new(WidgetId::new(0), "Agree");
        assert_eq!(checkbox.state, CheckboxState::Unchecked);
        checkbox.toggle();
        assert_eq!(checkbox.state, CheckboxState::Checked);
        checkbox.toggle();
        assert_eq!(checkbox.state, CheckboxState::Unchecked);
    }
}
