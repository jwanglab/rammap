
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    name: String,
    description: Option<String>,
    sequence: Vec<u8>,
    quality: Option<Vec<u8>>,
}

impl Record {
    pub fn new(name: String, description: Option<String>, sequence: Vec<u8>, quality: Option<Vec<u8>>) -> Self {
        Self { name, description, sequence, quality }
    }
    
    // Getters
    pub fn name(&self) -> &str { &self.name }
    pub fn description(&self) -> Option<&str> { self.description.as_deref() }
    pub fn sequence(&self) -> &[u8] { &self.sequence }
    pub fn quality(&self) -> Option<&[u8]> { self.quality.as_deref() }
}

/// Zero-copy view of a record
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefRecord<'a> {
    pub head: &'a [u8], // Full header line (without > or @)
    pub seq: &'a [u8],
    pub qual: Option<&'a [u8]>,
}

impl<'a> RefRecord<'a> {
    pub fn to_owned(&self) -> Record {
        // split head into name and desc
        let s = String::from_utf8_lossy(self.head);
        let mut parts = s.splitn(2, |c: char| c.is_whitespace());
        let name = parts.next().unwrap_or("").to_string();
        let description = parts.next().map(|s| s.to_string());
        
        Record {
            name,
            description,
            sequence: self.seq.to_vec(),
            quality: self.qual.map(|q| q.to_vec()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ref_record_to_owned() {
        let head = b"seq1 description comes here";
        let seq = b"ACGT";
        let qual = b"IIII";
        
        let r = RefRecord {
            head,
            seq,
            qual: Some(qual),
        };
        
        let owned = r.to_owned();
        assert_eq!(owned.name, "seq1");
        assert_eq!(owned.description, Some("description comes here".to_string()));
        assert_eq!(owned.sequence, b"ACGT");
        assert_eq!(owned.quality, Some(b"IIII".to_vec()));
    }

    #[test]
    fn test_ref_record_to_owned_no_desc() {
        let head = b"seq1";
        let r = RefRecord {
            head,
            seq: b"A",
            qual: None,
        };
        
        let owned = r.to_owned();
        assert_eq!(owned.name, "seq1");
        assert_eq!(owned.description, None);
    }
}
