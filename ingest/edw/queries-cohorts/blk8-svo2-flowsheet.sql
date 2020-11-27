SELECT DISTINCT
    t5.PatientIdentityID as MRN
    ,t1.PatientID
    ,t1.PatientEncounterID
    ,t1.DepartmentID
    ,t2.FlowsheetDataID
    ,t3.FlowsheetMeasureID
    ,t4.FlowsheetMeasureNM
    ,t4.DisplayNM
    ,t4.DisplayAbbreviationCD
    ,t3.RecordedDTS
    ,t3.EntryTimeDTS
    ,t3.MeasureTXT
    ,t3.MeasureCommentTXT
    ,t4.ValueTypeCD
    ,t4.ValueTypeDSC
    ,t4.UnitsCD
FROM
    Epic.Encounter.ADT_MGH t1
    INNER JOIN Epic.Clinical.FlowsheetRecordLink_MGH t2 ON t2.PatientID=t1.PatientID
    INNER JOIN Epic.Clinical.FlowsheetMeasure_MGH t3 ON t3.FlowsheetDataID = t2.FlowsheetDataID
    INNER JOIN Epic.Clinical.FlowsheetGroup_MGH t4 ON t3.FlowsheetMeasureID = t4.FlowsheetMeasureID
    INNER JOIN Epic.Patient.Identity_MGH t5 ON t5.PatientID=t1.PatientID AND t5.IdentityTypeID=67
WHERE
    t1.DepartmentID = '10020010623'
    AND (LOWER(t4.DisplayAbbreviationCD) = 'svo2' OR LOWER(t4.FlowsheetMeasureNM) = 'r phs ip mixed venous o2 for fick calculation')
