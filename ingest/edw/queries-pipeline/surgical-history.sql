SELECT DISTINCT
    t2.PatientIdentityID as MRN
    ,t1.PatientID
    ,t1.PatientEncounterID
    ,t3.LineNBR
    -- ,t3.ContactDTS
    ,t3.ProcedureID
    ,t4.ProcedureCD
    ,t4.ProcedureNM
    ,replace(replace(t3.CommentTXT,';',' -'), ',' ,' -') as CommentTXT
    ,replace(replace(t3.ProcedureCommentTXT,';',' -'), ',' ,' -') as ProcedureCommentTXT
    ,replace(replace(t3.SurgicalHistoryDateTXT,';',' -'), ',' ,' -') as SurgicalHistoryDateTXT
    ,t3.SurgicalHistoryStartDTS
FROM
    [Epic].[Encounter].[ADT_MGH] t1
    left join Epic.Patient.Identity_MGH t2 on t2.PatientID=t1.PatientID
    left join Epic.Patient.SurgicalHistory_MGH t3 on t2.PatientID = t3.PatientID
    left join [Epic].[Reference].[Procedure] t4 on t3.[ProcedureID] = t4.[ProcedureID]
WHERE
    t1.PatientEncounterID in ({}) and t2.IdentityTypeID = 67
