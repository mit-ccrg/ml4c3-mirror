USE MUSE_Site0001

IF OBJECT_ID('all_ecgs', 'U') IS NOT NULL
  DROP TABLE all_ecgs;

SELECT DISTINCT
	a.PatientID AS PatientID,
	b.AcquisitionDateTime_DT AS ECG_datetime
INTO
	all_ecgs
FROM
	tstPatientDemographics AS a
INNER JOIN
	tstTestDemographics AS b
ON
	a.TestID = b.TestID

SELECT * FROM all_ecgs
