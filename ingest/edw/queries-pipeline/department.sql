SELECT DISTINCT
    t2.PatientIdentityID as MRN
    ,t1.PatientEncounterID
    ,t1.DepartmentDSC
FROM
    Epic.Encounter.ADT_MGH t1
    left join Epic.Patient.Identity_MGH t2 on (t1.PatientID=t2.PatientID)
WHERE
    t2.IdentityTypeID = 67
    and t1.DepartmentID in ({})
