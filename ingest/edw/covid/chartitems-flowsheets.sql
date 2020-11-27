--- Flowsheet chart-items query for COVID data ---

    SELECT MRN,PatientID,PatientEncounterID,table2.InpatientDataID, table2.DepartmentID, FlowsheetDataID,FlowsheetMeasureID,FlowsheetMeasureNM,RecordedDTS,MeasureTXT
    FROM
    (
      SELECT  distinct t5.InpatientDataID,t4.FlowsheetDataID,t4.FlowsheetMeasureID,t4.RecordedDTS,t4.MeasureTXT,t6.FlowsheetMeasureNM
      FROM [Epic].[Clinical].[FlowsheetMeasure_MGH] t4
      INNER join Epic.Clinical.FlowsheetRecordLink_MGH t5 on t4.FlowsheetDataID=t5.FlowsheetDataID
      INNER join Epic.Clinical.FlowsheetGroup_MGH t6 on t4.FlowsheetMeasureID=t6.FlowsheetMeasureID
    WHERE
    t4.FlowsheetMeasureID
    in
    ('3042300800'
    ,'29597'
    ,'1040129598'
    ,'301360'
    ,'301570'
    ,'8'
    ,'9'
    ,'301250'
    ,'301260'
    ,'10'
    ,'301550'
    ,'301620'
    ,'401001'
    ,'61'
    ,'6'
    ,'301640'
    ,'10'
    ,'11'
    ,'14'
    ,'5'
    )

      ) table1
      inner join
      (
      SELECT  distinct t2.PatientID, t2.DepartmentID, InpatientDataID,PatientEncounterID,PatientIdentityID as MRN
      FROM [Epic].[Clinical].[InpatientDataStore_MGH] t1
      INNER join Epic.Encounter.ADT_MGH t2 on (t1.GenericPatientDatabaseCSNID=t2.PatientEncounterID)
      INNER join Epic.Patient.Identity_MGH t3 on (t2.PatientID=t3.PatientID and IdentityTypeID=67)
      where PatientIdentityID in ({})

      ) table2
      on table1.InpatientDataID=table2.InpatientDataID
