import pydicom
import tifffile

dicom_image = pydicom.dcmread("E:\MA_Data\MeasuredData\Synbone04-projections.dcm")
pixel_data = dicom_image.pixel_array
metadata = {str(tag): str(value) for tag, value in dicom_image.items() if tag != 0x7fe00010}

tifffile.imwrite("E:\MA_Data\MeasuredData\Synbone04-projections.tiff", pixel_data, imagej=True, metadata=metadata)