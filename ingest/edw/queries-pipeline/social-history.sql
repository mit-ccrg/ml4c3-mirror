SELECT DISTINCT
    t3.PatientIdentityID as MRN
    ,t1.[PatientID]
    ,t1.[PatientEncounterID]
    ,t2.TobaccoPacksPerDayCNT
    ,t2.TobaccoUsageYearNBR
    ,replace(replace(t2.TobaccoCommentTXT,';',' -'), ',' ,' -') as TobaccoCommentTXT
    ,t2.SmokingQuitDTS
    ,t2.TobaccoUserCD
    ,t2.TobaccoUserDSC
    ,t2.AlcoholOuncesPerWeekCNT
    ,replace(replace(t2.AlcoholCommentTXT,';',' -'), ',' ,' -')  as AlcoholCommentTXT
    ,t2.AlcoholUseCD
    ,t2.AlcoholUseDSC
    ,t2.AlcoholFrequencyCD
    ,t2.AlcoholFrequencyDSC
FROM
    [Epic].[Encounter].[ADT_MGH] t1
    left join Epic.Patient.SocialHistory_MGH t2 on t1.PatientID=t2.PatientID
    left join Epic.Patient.Identity_MGH t3 on t3.PatientID=t1.PatientID
WHERE
    t1.PatientEncounterID in ({}) and t3.IdentityTypeID = 67
