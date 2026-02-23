use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub timestamp: u64,
    pub content: String,
    pub metadata: HashMap<String, String>,
}

impl Snapshot {
    pub fn new(content: &str) -> Self {
        Self {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            content: content.to_string(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn content(&self) -> &str {
        &self.content
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotFormat {
    Raw,
    Json,
}

pub struct SnapshotTester {
    snapshots: HashMap<String, Snapshot>,
    format: SnapshotFormat,
}

impl SnapshotTester {
    pub fn new() -> Self {
        Self {
            snapshots: HashMap::new(),
            format: SnapshotFormat::Raw,
        }
    }

    pub fn with_format(mut self, format: SnapshotFormat) -> Self {
        self.format = format;
        self
    }

    pub fn record(&mut self, name: &str, content: &str) -> Snapshot {
        let snapshot = Snapshot::new(content);
        self.snapshots.insert(name.to_string(), snapshot.clone());
        snapshot
    }

    pub fn record_with_metadata(
        &mut self,
        name: &str,
        content: &str,
        metadata: HashMap<String, String>,
    ) -> Snapshot {
        let snapshot = Snapshot::new(content).with_metadata_multi(metadata);
        self.snapshots.insert(name.to_string(), snapshot.clone());
        snapshot
    }

    pub fn get(&self, name: &str) -> Option<&Snapshot> {
        self.snapshots.get(name)
    }

    pub fn compare(&self, name: &str, content: &str) -> bool {
        self.snapshots
            .get(name)
            .map(|s| s.content == content)
            .unwrap_or(false)
    }

    pub fn diff(&self, name: &str, content: &str) -> Option<SnapshotDiff> {
        self.snapshots.get(name).map(|snapshot| SnapshotDiff {
            name: name.to_string(),
            expected: snapshot.content.clone(),
            actual: content.to_string(),
        })
    }

    pub fn serialize(&self) -> Result<String, Box<dyn std::error::Error>> {
        match self.format {
            SnapshotFormat::Raw => {
                let mut output = String::new();
                for (name, snapshot) in &self.snapshots {
                    output.push_str(&format!("# {}\n", name));
                    output.push_str(&snapshot.content);
                    output.push('\n');
                }
                Ok(output)
            }
            SnapshotFormat::Json => serde_json::to_string_pretty(&self.snapshots)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>),
        }
    }

    pub fn deserialize(&mut self, data: &str) -> Result<(), Box<dyn std::error::Error>> {
        match self.format {
            SnapshotFormat::Raw => {
                let mut current_name = String::new();
                let mut current_content = String::new();

                for line in data.lines() {
                    if line.starts_with("# ") {
                        if !current_name.is_empty() {
                            self.snapshots
                                .insert(current_name.clone(), Snapshot::new(&current_content));
                        }
                        current_name = line[2..].to_string();
                        current_content.clear();
                    } else {
                        current_content.push_str(line);
                        current_content.push('\n');
                    }
                }

                if !current_name.is_empty() {
                    self.snapshots
                        .insert(current_name, Snapshot::new(&current_content));
                }
                Ok(())
            }
            SnapshotFormat::Json => {
                self.snapshots = serde_json::from_str(data)
                    .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
                Ok(())
            }
        }
    }
}

impl Default for SnapshotTester {
    fn default() -> Self {
        Self::new()
    }
}

impl Snapshot {
    pub fn with_metadata_multi(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

#[derive(Debug)]
pub struct SnapshotDiff {
    pub name: String,
    pub expected: String,
    pub actual: String,
}

impl fmt::Display for SnapshotDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Snapshot '{}' mismatch:\n", self.name)?;
        write!(f, "Expected:\n{}\n", self.expected)?;
        write!(f, "Actual:\n{}\n", self.actual)
    }
}

pub struct TestHarness {
    width: u16,
    height: u16,
    tester: SnapshotTester,
}

impl TestHarness {
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            tester: SnapshotTester::new(),
        }
    }

    pub fn with_snapshots(mut self, tester: SnapshotTester) -> Self {
        self.tester = tester;
        self
    }

    pub fn size(&self) -> (u16, u16) {
        (self.width, self.height)
    }

    pub fn tester(&mut self) -> &mut SnapshotTester {
        &mut self.tester
    }

    pub fn snapshot(&mut self, name: &str, content: &str) {
        self.tester.record(name, content);
    }

    pub fn assert_snapshot(&self, name: &str, content: &str) -> Result<(), SnapshotDiff> {
        match self.tester.diff(name, content) {
            Some(diff) => Err(diff),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snapshot_new() {
        let snapshot = Snapshot::new("Hello, World!");
        assert_eq!(snapshot.content(), "Hello, World!");
    }

    #[test]
    fn test_snapshot_with_metadata() {
        let snapshot = Snapshot::new("content").with_metadata("key", "value");

        assert_eq!(snapshot.content(), "content");
        assert_eq!(snapshot.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_snapshot_tester_record() {
        let mut tester = SnapshotTester::new();
        tester.record("test", "content");

        assert!(tester.get("test").is_some());
        assert_eq!(tester.get("test").unwrap().content(), "content");
    }

    #[test]
    fn test_snapshot_tester_compare() {
        let mut tester = SnapshotTester::new();
        tester.record("test", "content");

        assert!(tester.compare("test", "content"));
        assert!(!tester.compare("test", "other"));
    }

    #[test]
    fn test_snapshot_tester_serialize_raw() {
        let mut tester = SnapshotTester::new();
        tester.record("test1", "content1");
        tester.record("test2", "content2");

        let output = tester.serialize().unwrap();
        assert!(output.contains("# test1"));
        assert!(output.contains("content1"));
    }

    #[test]
    fn test_snapshot_tester_serialize_json() {
        let mut tester = SnapshotTester::new().with_format(SnapshotFormat::Json);
        tester.record("test", "content");

        let output = tester.serialize().unwrap();
        assert!(output.contains("test"));
        assert!(output.contains("content"));
    }

    #[test]
    fn test_test_harness_new() {
        let harness = TestHarness::new(80, 24);
        assert_eq!(harness.size(), (80, 24));
    }

    #[test]
    fn test_snapshot_diff_display() {
        let diff = SnapshotDiff {
            name: "test".to_string(),
            expected: "expected".to_string(),
            actual: "actual".to_string(),
        };

        let display = format!("{}", diff);
        assert!(display.contains("test"));
        assert!(display.contains("expected"));
        assert!(display.contains("actual"));
    }
}
