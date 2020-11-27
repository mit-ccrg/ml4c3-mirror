--filter by ContactDTS in Python if want to only include diagnoses existing on or before AdmitDate
SELECT
distinct 'null',
id.PatientIdentityID
,ContactDTS
,hist.DiagnosisID
,ref.DiagnosisNM

FROM

Epic.Patient.MedicalHistory_MGH hist
INNER JOIN Epic.Reference.ICDDiagnosis ref on hist.DiagnosisID = ref.DiagnosisID
INNER JOIN Epic.Patient.Identity_MGH id ON hist.PatientID=id.PatientID

WHERE id.PatientIdentityID in ()
AND IdentityTypeID IN (67,227)
