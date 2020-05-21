# IPBSD
Integrated Performance-Based Seismic Design

Design framework based on limiting economic losses, i.e. expected annual loss (EAL) and targetting a probability of collapse, i.e. mean annual frequency of collapse (MAFC).

**Contents**<a id='contents'></a>
1. [Input arguments and files](#input)
2. [Step by step procedures](#process)
3. [Future upgrade objectives](#future)


### Input arguments and files <a id='input'>

Additional explanations of each file are provided within the relevant directories.

**Input arguments are**
1. **Limit EAL** - Limiting economic loss value in terms of expected annual loss

2. **Target MAFC** - Target collapse safety value in terms of mean annual frequency of exceedance

3. **Damping** - Ratio of critical damping (default 0.05)

4. **Analysis type** - Type of analysis for the definition of demands on the structural elements<br/>
	1. Simplified ELF -        	No analysis is run, calculations based on simplified expressions<br /> 
	2. ELF -                  	Equivalent lateral force method of analysis<br/>
	3. ELF & gravity -       	Analysis under ELF and gravity loads<br/>
	4. RMSA -                	Response method of spectral analysis<br/.
	5. RMSA & gravity -      	Analysis under RMSA and gravity loads<br/>
5. **input.csv** - input file comprising of the following arguments:<br/>
    - design_scenario -       		tag/id of case to be used to store results in \database<br/>
    - PLS -                   		performance limit states of interest<br/>
    - ELR -                    		expected loss ratios corresponding to PLS<br/>
    - TR -                     		return periods corresponding to PLS<br/>
    - aleatory -               		aleatory uncertainties corresponding to PLS<br/>
    - SLF -                  		storey loss functions (linear, nonlinear or provided)<br/>
    - bldg_ch -               		building characteristics (floor loading in kPa, roof loading in kPa, floor area)<br/>
    - h_storeys -             		heights of stories<br/>
    - mode_red -              		higher mode reduction factor<br/>
    - PFA_convert -            		peak floor acceleration conversion factor<br/>
    - spans_X -               		bay widths<br/>
6. **slf.xlsx** - Storey-loss-functions file<br/>
7. **hazard.pickle** - Contains information on<br/>
				a) intensity measure (IM) levels<br/>
				b) Spectral acceleration range associated with each IM<br/>
				c) Annual probability of exceedance list associated with each IM e.g. [['PGA',''SA(0.1)', ...], [sa1_list, sa2_list, ...], [apoe1_list, apoe2_list, ...]]<br/>
8. **spo.csv** - Static pushover curve parameter assumptions<br/>
                            	Currently used as input for SPO2IDA<br/>
                            	Features:<br/>
	- mc - 			Hardening ductility<br/>
	- a - 			Hardening slope<br/>
	- ac - 			Softening slope<br/>
	- r -			Residual strength ratio with respect to yield strength<br/>
	- mf -			Fracturing ductility<br/>
	- pw - 			Pinching weight (default 1.0)<br/>
    
</a><font color=blue><div style="text-align: right">[up](#contents)

**Step by step procedure** (#process)

-> **Phase 1 - Performance objectives:**<br/>

		1. Define limit EAL (economic loss) and target MAFC (collapse safety) - Input
		and supply other input arguments; methods: read_inputs<br/>
		2. Supply seismic hazard and perform second-order fitting - Input, Hazard<br/>
			a. Mean annual frequency of exceeding performance limit states<br/>
			b. Peak ground acceleration at limit states relevant for EAL (e.g. SLS)
			methods: read_hazard<br/>
		3. Get the loss curve - LossCurve, EALCheck<br/>
			a. EAL calculated<br/>
			b. EAL verification<br/>
		* if EAL < EAL limit, continue, otherwise update performance objectives in Step 3<br/>
		4. Supply storey loss functions, or relevant parameters (e.g. HAZUS) - SLF
		and Get design engineering demand parameters (EDPs) - DesignLimits<br/>
			a. Peak storey drift, PSD<br/>
			b. Peak floor acceleration, PFA<br/>
		5. Perform design to spectral value transformations of EDPs - Transformations<br/>
			a. Effective first mode mass<br/>
			b. First mode participation factor<br/>
			c. Design spectral displacement<br/>
			d. Design spectral acceleration<br/>
-> **Phase 2 - Building information:** <br/>

		1. Get design spectra at limit states relevant for EAL (e.g. SLS) - Spectra<br/>
			a. Sa and Sd<br/>
		2. Get feasible period range - PeriodRange<br/>
			a. Lower period bound<br/>
			b. Upper period bound<br/>
		3. Optimization for fundamental period to identify all possible structural solutions within the period range -
		CrossSection, GetT1<br/>
			a. All solutions meeting the period range condition<br/>
			b. Optimal solution based on least weight<br/>
-> **Phase 3 - Collapse safety consideration:** <br/>
		1. Identify possible static pushover curve (SPO) information as input for SPO2IDA - Input<br/>
		2. Perform SPO2IDA and derive collapse fragility function<br/>
		    a. IDA curves<br/>
		    b. Fractiles of R at collapse<br/>
		3. Perform optimization for MAFC - MAFCCheck<br/>
            a. Spectral acceleration at yield<br/>
            b. Yield displacement<br/>
-> **Phase 4 - Design:**<br/>
        1. Identify design actions on the structure - Action<br/>
            a. Lateral forces<br/>
        2. Perform ELFM and identify demands on the structural members - OpenSeesRun<br/>
            a. Demands<br/>
        3. Perform moment-curvature sectional analysis to identify the necessary reinforcement, includes capacity design
           requirements per Eurocode - MPhi, Detailing<br/>
            a. Reinforcement ratios<br/>
            b. Moment capacities<br/>
            c. Curvature ductility<br/>
            d. Cracked section propertis<br/>
-> **Phase 5 - Detailing:**<br/>
        1. Use the optimal solution and estimate Period based on cracked section properties of 4.3d - Detailing, CrossSection<br/>
        	a. Fundamental period<br/>
        	b. Verify that the period is within the bounds<br/>
        2. Estimate system hardening ductility - Detailing<br/>
        	a. System hardening ductility<br/>
        ***3. todo, add estimations of other parameters of SPO curve***<br/>
        X. Check for conditions met, if non met, return to 3.1 and repeat until conditions are met - Detailing<br/>
-> ***Phase 6 - Modifications and Rerunning of Phases if necessary:***<br/>
        1.<br/>

</a><font color=blue><div style="text-align: right">[up](#contents)
  
**Future upgrade objectives** (#future)

* All input xlsx, csv etc. files will be modified to be more flexible
* Add explanations on how and in which format to provide the inputs for the software
* Reinforced concrete moment-resisting frame -> Steel MRF, other typologies to be included
* 3 Performance objective limit states -> flexible for inclusion
* Homogeneous storey-loss-functions along the height -> flexible for inclusion
* SPO2IDA tool for collapse fragility definition -> ML algorithms to avoid using SPO2IDA, needs extensive studies and analysis, possible data collection
* Symmetric structures only -> add considerations for 3D models
* Single conversion factor for peak floor accelerations -> study based on Antonio's work to include regressions
* Code-based overstrength factors indirectly accounted for -> flexible to include
* Same beam cross-sections along the height/no grouping for beams -> add grouping once generic opensees model is updated to account for it
* Considerations for shear design, will be useful also in identifying inelastic rotation capacities

</a><font color=blue><div style="text-align: right">[up](#contents)
  
