SELECT DISTINCT
    t5.PatientIdentityID AS MRN
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
    LEFT JOIN Epic.Encounter.ADT_MGH t6 ON t1.LastInADTEventID=t6.EventID
    LEFT JOIN Epic.Clinical.FlowsheetRecordLink_MGH t2 ON t2.PatientID=t1.PatientID
    LEFT JOIN Epic.Clinical.FlowsheetMeasure_MGH t3 ON t3.FlowsheetDataID = t2.FlowsheetDataID
    LEFT JOIN Epic.Clinical.FlowsheetGroup_MGH t4 ON t3.FlowsheetMeasureID = t4.FlowsheetMeasureID
    LEFT JOIN Epic.Patient.Identity_MGH t5 ON t5.PatientID=t1.PatientID
WHERE
    t1.PatientEncounterID IN ({})
    AND t5.IdentityTypeID=67
    AND t3.RecordedDTS BETWEEN t6.EffectiveDTS AND t1.EffectiveDTS
