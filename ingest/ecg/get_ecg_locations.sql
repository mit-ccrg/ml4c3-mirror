USE MUSE_Site0001

IF OBJECT_ID('ecg_locations', 'U') IS NOT NULL
  DROP TABLE ecg_locations

SELECT DISTINCT
	a.PatientID AS PatientID,
	b.AcquisitionDateTime_DT AS ECG_datetime,
	b.Location AS LocationID,
	c.FullName AS LocationName
INTO
	ecg_locations
FROM
	tstPatientDemographics AS a
INNER JOIN
	tstTestDemographics AS b
ON
	a.TestID = b.TestID
INNER JOIN
	cfgLocations AS c
ON
	b.Location = c.LocationID


SELECT * FROM ecg_locations
