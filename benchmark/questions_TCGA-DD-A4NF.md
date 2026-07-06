# Benchmark Questions: TCGA-DD-A4NF (liver HCC)

A single patient with all three modalities, used to test the agent end to end.

**Data:** Radiology (4 CT, ~47 MR) · Pathology (1 whole-slide image) · Clinical (5 notes + path report PDF)

The categories go from easy (find one fact) to hard (fuse modalities, reason over time, resolve conflicts).

## 1. Retrieval
Pull one fact from one modality. The floor.

- Diagnosis and ICD-O morphology code?
- Which liver segment is the tumor in?
- LI-RADS category and the features supporting it?
- Surgical margin status, vascular invasion, and grade?
- Treatment at recurrence, dose, and response?
- Body part and contrast agent for the primary contrast CT?
- KVP, slice thickness, kernel, and patient position for CT_Series-15340?
- WSI level count, level-0 dimensions, and microns-per-pixel?

## 2. Cross-modal
Carry a fact from one modality into another.

- Which MR series is the multiphase liver MRI behind the LR-5 read?
- Does the pathology grade agree with the clinical assessment?
- Do the slide barcode, DICOM PatientID, and clinical ID all match?
- Is there imaging consistent with the 2013 recurrence (local + bone)?

## 3. Temporal
Order events and compute intervals across notes.

- Reconstruct the timeline: diagnosis, resection, recurrence, radiation, last contact.
- How long after resection did recurrence occur?

## 4. Conflict-resolution
Notice disagreement and pick the right source. The notes are seeded with contradictions.

- Tumor diameter: voice-rec says 4.0, attending corrected to 3.87.
- Patient age: note says 72, chart says 71.
- Liver MRI date: 2004 is flagged unreliable and clashes with the 2011 course.

## 5. Grounding / abstention
Say "not documented" instead of inventing, and cite the source.

- What is the HCV genotype? (not documented)
- What is the family history of cancer? (none)
- Is there portal vein thrombosis or ascites? (neither)
- Is the patient a transplant candidate? (no)
- What is the AFP, and is it elevated? (6.0, borderline)

## 6. Perception (optional)
Read the pixels, not just the headers. ABRA shows this is where agents fail most.

- Is arterial hyperenhancement visible in the tumor on MRI?
- Indicate the tumor-bearing region on the WSI thumbnail.
