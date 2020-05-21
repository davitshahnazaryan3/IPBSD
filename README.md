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

        1. Simplified ELF -        	No analysis is run, calculations based on simplified expressions
        2. ELF -                  	Equivalent lateral force method of analysis
        3. ELF & gravity -       	Analysis under ELF and gravity loads
        4. RMSA -                	Response method of spectral analysis
        5. RMSA & gravity -      	Analysis under RMSA and gravity loads
		
5. **input.csv** - input file comprising of the following arguments:<br/>

    	design_scenario -       	tag/id of case to be used to store results in \database
    	PLS -                   	performance limit states of interest
    	ELR -                    	expected loss ratios corresponding to PLS
    	TR -                     	return periods corresponding to PLS
        aleatory -               	aleatory uncertainties corresponding to PLS
        SLF -                  		storey loss functions (linear, nonlinear or provided)
        bldg_ch -               	building characteristics (floor loading in kPa, roof loading in kPa, floor area)
        h_storeys -             	heights of stories
        mode_red -              	higher mode reduction factor
        PFA_convert -            	peak floor acceleration conversion factor
        spans_X -               	bay widths
	
6. **slf.xlsx** - Storey-loss-functions file<br/>

7. **hazard.pickle** - Contains information on<br/>

        a) intensity measure (IM) levels
        b) Spectral acceleration range associated with each IM
        c) Annual probability of exceedance list associated with each IM,
        e.g. [['PGA',''SA(0.1)', ...], [sa1_list, sa2_list, ...], [apoe1_list, apoe2_list, ...]]
				
8. **spo.csv** - Static pushover curve parameter assumptions<br/>

        Currently used as input for SPO2IDA
        Features:
        - mc -                    	Hardening ductility
        - a -                    	Hardening slope
        - ac -                    	Softening slope
        - r -                    	Residual strength ratio with respect to yield strength
        - mf -                    	Fracturing ductility
        - pw -                    	Pinching weight (default 1.0)
    
</a><font color=blue><div style="text-align: right">[up](#contents)

**Step by step procedure**<a id='process'>

-> **Phase 1 - Performance objectives:**<br/>

		1. Define limit EAL (economic loss) and target MAFC (collapse safety) - Input
		and supply other input arguments; methods: read_inputs
		2. Supply seismic hazard and perform second-order fitting - Input, Hazard
			a. Mean annual frequency of exceeding performance limit states
			b. Peak ground acceleration at limit states relevant for EAL (e.g. SLS)
			methods: read_hazard
		3. Get the loss curve - LossCurve, EALCheck
			a. EAL calculated
			b. EAL verification
		* if EAL < EAL limit, continue, otherwise update performance objectives in Step 3
		4. Supply storey loss functions, or relevant parameters (e.g. HAZUS) - SLF
		and Get design engineering demand parameters (EDPs) - DesignLimits
			a. Peak storey drift, PSD
			b. Peak floor acceleration, PFA
		5. Perform design to spectral value transformations of EDPs - Transformations
			a. Effective first mode mass
			b. First mode participation factor
			c. Design spectral displacement
			d. Design spectral acceleration
			
-> **Phase 2 - Building information:** <br/>

		1. Get design spectra at limit states relevant for EAL (e.g. SLS) - Spectra
			a. Sa and Sd
		2. Get feasible period range - PeriodRange
			a. Lower period bound
			b. Upper period bound
		3. Optimization for fundamental period to identify all possible structural solutions within the period range -
		CrossSection, GetT1
			a. All solutions meeting the period range condition
			b. Optimal solution based on least weight
			
-> **Phase 3 - Collapse safety consideration:** <br/>

		1. Identify possible static pushover curve (SPO) information as input for SPO2IDA - Input
		2. Perform SPO2IDA and derive collapse fragility function
		    	a. IDA curves<br/>
		    	b. Fractiles of R at collapse
		3. Perform optimization for MAFC - MAFCCheck
            		a. Spectral acceleration at yield
            		b. Yield displacement
-> **Phase 4 - Design:**<br/>

        	1. Identify design actions on the structure - Action
           		a. Lateral forces
        	2. Perform ELFM and identify demands on the structural members - OpenSeesRun
            		a. Demands
        	3. Perform moment-curvature sectional analysis to identify the necessary reinforcement, includes capacity design
           		requirements per Eurocode - MPhi, Detailing
            		a. Reinforcement ratios
            		b. Moment capacities
            		c. Curvature ductility
            		d. Cracked section propertis
			
-> **Phase 5 - Detailing:**<br/>

        	1. Use the optimal solution and estimate Period based on cracked section properties of 4.3d - Detailing,
			CrossSection
        		a. Fundamental period
        		b. Verify that the period is within the bounds
        	2. Estimate system hardening ductility - Detailing
        		a. System hardening ductility
        	***3. todo, add estimations of other parameters of SPO curve***<br/>
        	X. Check for conditions met, if non met, return to 3.1 and repeat until conditions are met - Detailing
		
-> ***Phase 6 - Modifications and Rerunning of Phases if necessary:***<br/>

        	1.

</a><font color=blue><div style="text-align: right">[up](#contents)
  
**Future upgrade objectives**<a id='future'>

* [ ] All input xlsx, csv etc. files will be modified to be more flexible

* [ ] Add explanations on how and in which format to provide the inputs for the software

* [ ] Reinforced concrete moment-resisting frame -> Steel MRF, other typologies to be included

* [ ] 3 Performance objective limit states -> flexible for inclusion

* [ ] Homogeneous storey-loss-functions along the height -> flexible for inclusion

* [ ] SPO2IDA tool for collapse fragility definition -> ML algorithms to avoid being limited to using SPO2IDA, needs extensive studies and analysis, possible data collection

* [ ] Symmetric structures only -> add considerations for 3D models

* [ ] Single conversion factor for peak floor accelerations -> study based on Antonio's work to include regressions

* [ ] Code-based overstrength factors indirectly accounted for -> flexible to include

* [ ] Same beam cross-sections along the height/no grouping for beams -> add grouping once generic opensees model is updated to account for it

* [ ] Considerations for shear design, will be useful also in identifying inelastic rotation capacities


</a><font color=blue><div style="text-align: right">[up](#contents)
  
