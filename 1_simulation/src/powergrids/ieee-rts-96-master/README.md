# IEEE RTS-96

The IEEE Reliability Test System-1996 is a benchmark system for power grids. It consists of 73 buses and 108 lines with 33 generators.

In *Data* I provide .csv files for conveniently building the test cases in Julia.

| File | Data |
|------|------|
| *Bus.csv* | Number, ID, Names, Type & Base Voltage of Buses |
| *Line.csv* | Source / Destination Buses, Reactances & Resistances of Lines |
| *Load.csv* | Bus ID, Active & Reactive Power of Loads |
| *Generator.csv* | Bus ID, Voltage Setpoints and Active Power of Generators |
| *Flow.csv* | Voltage Angles and Magnitudes of Power Flow Solution |
