SELECT distinct 'null',t2.PatientIdentityID as MRN
      ,[BirthDTS]
      ,[DeathDTS]
      ,[SexDSC]
	  ,[EthnicGroupDSC]
	  ,race.PatientRaceDSC
	  ,[ZipCD]
FROM [Epic].[Patient].[Patient_MGH] t1
inner join epic.Patient.Identity_MGH t2 on  (t1.PatientID=t2.PatientID  and t2.IdentityTypeID=67 )
inner join [Epic].[Patient].Race_MGH race on race.PatientID=t1.PatientID
where t2.PatientIdentityID in ()
