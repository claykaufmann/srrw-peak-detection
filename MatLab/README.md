## Functions
### Clean.m / Clean.mlx
(Cleam.mlx is a 'python notebook' style format and is much easier to work with. It is also where all recent work has been done: Clean.m is not up to date). 
These are the steps in clean.m to detect and, if applicable, correct anomalies in the DB-SRRW time series using HydRun.
1. Data are loaded from **converted_data/**, and event onsets and offsets are marked by hydrun. 
2. HydRun returns the data in the form of a one column table of `[event<link>]`, where the link is a reference to a two column table of `[sample time, sample value]`, where the sample time is a Julian time and the sample value is the amplitude of the sensor reading at the sample time.
3. The marked fDOM events are checked for anomalies, and labelled by the event type (one of `[valid, tilted, 3peak, invalid]`, where all types other than `valid` represent the anomaly types considered in this work). (`Titled` means a tilted peak; `3peak` means violation of the three peak order among stage, turbidity, and fDOM; `invalid` all the other anomaly types (see the research outline document). As a result, the `[event<link>]` table is then augmented to `[event<link>, event_type]`, so every event is labelled with its event type.
4. The `[event<link>, event_type]` table is then passed to a function that either corrects the anomaly (if not `valid`) (as of now, cut out the anomalous times series segement using linear interpolation between onset and offset) or flags it as such, depending on what the user wants for the particular anomaly type.
5. The results below show the raw vs. corrected fDOM on top, stage in the middle, and turbidity at the bottom. HydRun was used on all three types of time series (i.e., fDOM, stage, turbidity), but the figure belows shows HydRun output only for the stage.

### Auxiliary functions of clean.m
These are auxiliary functions implemented to perform the steps described above in the Usage section.  There is a more detailed description of each function in the header of the function code.
1. `flag_runoffs` takes runoff events as generated by HydRun and flags them based on our anomaly detection rules.
2. `flip_tseries` flips a timeseries upside down, and is used for removing negative spikes.
3. `interp_flagged_events` interpolates accross event onset to offset located by HydRun; this "cuts out" the invalid events.
4. `interp_tseries` interpolates across event onset to offset given as inputs to this function (e.g., as of now called by interp_flagged_events).
5. `juliansloperatio` computes the slope ratio of an event in Julian time scale.
6. `plotflaggedrunoffevent` is an extension to `HydRun/plotrunoffevent`, which flags the anomaly type on screen when plotting the time series.
7. `string2juldate` converts a date string to Julian time.
8. `tseries_to_mat` pares down original csv tables and converts it to **.mat** files for easy loading.
