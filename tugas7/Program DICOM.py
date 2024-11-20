import pydicom
from pydicom.dataset import Dataset, FileDataset
import numpy as np
import cv2
import datetime

# Fungsi untuk membuat dan menyimpan file DICOM dari citra yang diunggah
def create_dicom_from_image(image_path, output_filename):
    # Baca citra menggunakan OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Gagal membaca citra. Pastikan path file citra benar.")
        return

    # Konversi citra menjadi format uint16 (DICOM biasanya menggunakan format ini)
    image = image.astype(np.uint16)

    # Buat dataset DICOM baru
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.1')  # SOP Class UID
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(output_filename, {}, file_meta=file_meta, preamble=b'\0' * 128)

    # Tambahkan metadata standar
    ds.PatientName = "Phantom^Citra"
    ds.PatientID = "123456"
    ds.Modality = "OT"  # Other
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Tambahkan data citra
    ds.PixelData = image.tobytes()
    ds.Rows, ds.Columns = image.shape
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0  # Unsigned integer

    # Tambahkan timestamp
    ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')

    # Simpan file DICOM
    pydicom.dcmwrite(output_filename, ds)
    print(f"DICOM file '{output_filename}' berhasil dibuat.")

# Path ke file citra input (ubah path ini dengan file citra phantom Anda)
image_path = 'C:/Users/Lenovo/Pictures/Screenshots/Screenshot 2024-11-05 132940.png'  # Ganti dengan path file gambar Anda
output_filename = 'bar.dcm'

create_dicom_from_image(image_path, output_filename)
