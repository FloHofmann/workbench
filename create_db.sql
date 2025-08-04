CREATE TABLE schematic (
	AnimalId varchar(255),
	CellId int,
	Condition blob DEFAULT '-',
	Folderpath blob DEFAULT '-',
	use bool DEFAULT 0,
	PRIMARY KEY (AnimalId, CellId)
);

