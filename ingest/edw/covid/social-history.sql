SELECT 'null',
	PatientIdentityID,SmokingTobaccoUseDSC, AlcoholUseDSC, --SmokingTobaccoUseDSC and AlcoholUseDSC are most useful fields - other fields included for completeness
	TobaccoPacksPerDayCNT,TobaccoUsageYearNBR,TobaccoCommentTXT,ContactDTS,
	TobaccoUserDSC,SmokelessTobaccoUserDSC,
	SmokingStartDTS,SmokingQuitDTS,
	CigarettesFLG,PipesFLG,CigarsFLG,SnuffFLG,ChewFLG,
	AlcoholFrequencyDSC,AlcoholBingeDSC,AlcoholDrinkPerDayDSC,
	AlcoholOuncesPerWeekCNT,AlcoholCommentTXT,days_elapsed
FROM
(SELECT
	PatientIdentityID, TobaccoPacksPerDayCNT,TobaccoUsageYearNBR,TobaccoCommentTXT,ContactDTS,
	TobaccoUserDSC,SmokelessTobaccoUserDSC,SmokingTobaccoUseDSC,
	SmokingStartDTS,SmokingQuitDTS,
	CigarettesFLG,PipesFLG,CigarsFLG,SnuffFLG,ChewFLG,
	AlcoholUseDSC,AlcoholFrequencyDSC,AlcoholBingeDSC,AlcoholDrinkPerDayDSC,
	AlcoholOuncesPerWeekCNT,AlcoholCommentTXT,
	DATEDIFF(dd,CAST(pairs.admitdate as DATE),CAST(social.ContactDTS as DATE)) as days_elapsed,
	row_number() over (partition by id.PatientIdentityID order by DATEDIFF(dd,CAST(pairs.admitdate as DATE),CAST(social.ContactDTS as DATE)) desc) as rn
FROM
	Epic.Patient.SocialHistory_MGH social
	inner join Epic.Patient.Identity_MGH id on social.PatientID=id.PatientID and IdentityTypeID in (67,227)
	inner join (VALUES (),()) --pairs (MRN, AdmitDate)
AS pairs (mrn, admitdate)
  ON (mrn=id.PatientIdentityID)
  WHERE social.ContactDTS <= pairs.admitdate) t
WHERE rn=1

/***
Note: Patients have multiple records (recorded at different times) of smoking/alcohol history.
I took the values from the recent recorded date, even if the recorded date happened after admission, because some patients didn't have a
smoking/alcohol history until after they were admitted for COVID.
***/
