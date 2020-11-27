SELECT DISTINCT
    t3.PatientIdentityID as MRN
    ,t1.[PatientID]
    ,t1.[PatientEncounterID]
    ,t4.LineNBR
    ,t4.DiagnosisID
    ,replace(replace(t5.DiagnosisNM,';',' -'), ',' ,' -') as DiagnosisNM
    -- ,t4.ContactDTS
    -- ,t4.PatientEncounterDateRealNBR
    ,replace(replace(t4.CommentTXT,';',' -'), ',' ,' -') as CommentTXT
    ,t4.MedicalHistoryDateTXT
    ,t4.MedicalHistoryStartDTS
    -- ,t4.MedicalHistoryEndDTS
FROM
    [Epic].[Encounter].[ADT_MGH] t1
    left join Epic.Patient.Identity_MGH t3 on t3.PatientID=t1.PatientID
    left join Epic.Patient.MedicalHistory_MGH t4 on t3.PatientID=t4.PatientID
    left join [Epic].[Reference].[ICDDiagnosis] t5 on t4.DiagnosisID = t5.DiagnosisID
WHERE
    t1.PatientEncounterID in ({}) and t3.IdentityTypeID = 67
