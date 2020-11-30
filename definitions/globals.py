# Imports: third party
import pytz

TIMEZONE = pytz.timezone("US/Eastern")

EPS = 1e-7

PDF_EXT = ".pdf"
TENSOR_EXT = ".hd5"
MODEL_EXT = ".h5"
XML_EXT = ".xml"
CSV_EXT = ".csv"

MRN_COLUMNS = {
    "mrn",
    "mgh_mrn",
    "medrecn",
    "patient_id",
    "patientid",
    "sample_id",
    "sampleid",
}

YEAR_DAYS = 365.26
