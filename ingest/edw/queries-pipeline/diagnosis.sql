SELECT DISTINCT
    t2.PatientIdentityID as MRN
    ,t1.PatientEncounterID as PatientEncounterID
    ,t1.PatientClassDSC
    ,t1.PatientServiceDSC
    ,t3.ContactDTS
    ,t3.DiagnosisID
    ,replace(t4.DiagnosisNM,',',' -') as DiagnosisNM
    ,t6.ICD9
    ,replace(t6.ICD9DSC,',',' -') as ICD9DSC
    ,t6.ICD10
    ,t6.ICD10DSC
FROM
    Epic.Encounter.ADT_MGH t1
    left join Epic.Patient.Identity_MGH t2 on t1.PatientID=t2.PatientID
    left join Epic.Encounter.PatientEncounterDiagnosis_MGH t3 on (t1.PatientEncounterID=t3.PatientEncounterID and t3.DiagnosisID is not null)
    left join Epic.Reference.ICDDiagnosis t4 on t3.DiagnosisID = t4.DiagnosisID
    left join Epic.Reference.DiagnosisCurrentICD9 t5 on t3.DiagnosisID=t5.DiagnosisID
    left join Misc.Reference.ICD9toICD10Mapping t6 on t5.ICD9CD=t6.ICD9DottedCD
WHERE
    t1.PatientEncounterID in ({}) and t2.IdentityTypeID=67
