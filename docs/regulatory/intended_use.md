# Intended Use Statement (Draft)

**Product working name:** Medical Imaging Report  
**Status:** Research / pre-submission draft — not cleared or approved by CDSCO, FDA, or any other regulator.

## Intended use

The Medical Imaging Report software is intended to assist trained radiologists by:

1. Ingesting diagnostic imaging studies (DICOM CT/MRI and, for demonstration, PNG/JPG slices).
2. Highlighting candidate regions of interest with location, size, and confidence metadata.
3. Generating a structured preliminary report (JSON and FHIR `DiagnosticReport`) for human review.
4. Capturing clinician accept/reject feedback for quality monitoring and future model improvement.

## Intended users

Licensed radiologists and imaging clinicians operating under institutional protocols.

## Intended environment

Hospital or imaging-center radiology workflow, integrated with PACS/RIS/EHR after formal validation.

## Explicit non-intended uses

- Not a standalone diagnostic device.
- Not for use without licensed clinician review and sign-off.
- Not for emergency triage as the sole decision source.
- Not cleared for marketing as a medical device until regulatory authorization is obtained.

## Current operating modes

| Mode | Description | Clinical use |
|---|---|---|
| `demo` | Heuristic detector + PneumoniaMNIST CNN | Forbidden for clinical care |
| `clinical` | CT/MRI model adapters when registry-approved; heuristic fallback otherwise | Only after V&V + institutional approval + regulatory pathway completion |
