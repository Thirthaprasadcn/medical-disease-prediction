import io

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from backend.pipeline.dicom_ingest import ingest_dicom_bytes, is_dicom_bytes


def _make_dicom_bytes() -> bytes:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.PatientID = "DEMO123"
    ds.PatientName = "DEMO^PATIENT"
    ds.Rows = 32
    ds.Columns = 32
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1
    arr = np.full((32, 32), 0, dtype=np.int16)
    arr[10:20, 10:20] = 500
    ds.PixelData = arr.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def test_is_dicom_and_ingest():
    raw = _make_dicom_bytes()
    assert is_dicom_bytes(raw)
    result = ingest_dicom_bytes(raw, window_name="soft")
    assert result["slice"].shape == (32, 32)
    assert result["metadata"]["modality"] == "CT"
    assert result["metadata"]["patient_id"] == "DEMO123"
