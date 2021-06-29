# IPBSD
Integrated Performance-Based Seismic Design

Design framework based on limiting economic losses, i.e. expected annual loss (EAL) and targeting a probability of collapse, i.e. mean annual frequency of collapse (MAFC).
As long as there are no Failures and/or Warnings, then the framework has successfully completed.

**Contents**<a id='contents'></a>
1. [Literature](#lit)
2. [Input arguments and files](#input)
3. [Step by step procedures](#process)
4. [Future upgrade objectives](#future)

### Literature <a id='lit'>

* Shahnazaryan D, O’Reilly GJ, Monteiro R. Storey loss functions of seismic design and assessment: development of tools and application. *Earthquake Spectra* 2021; DOI: 10.1177/87552930211023523.
	
* Shahnazaryan D, O’Reilly GJ. Integrated expected loss and collapse risk in performance-based seismic design of structures. *Bulletin of Earthquake Engineering* 2021; **19**(2): 978-1025. DOI: 10.1007/s10518-020-01003-x.
	
* Shahnazaryan D, O’Reilly GJ, Monteiro R. Using direct economic losses and collapse risk for seismic 
design of RC buildings. *COMPDYN 2019 - 7th International Conference on Computational Methods in 
Structural Dynamics and Earthquake Engineering*, Crete Island, Greece: 2019. DOI: https://doi.org/10.7712/120119.7281.19516.

* O’Reilly GJ, Calvi GM. Conceptual seismic design in performance-based earthquake engineering. 
*Earthquake Engineering & Structural Dynamics* 2019; **48**(4): 389–411. DOI: 10.1002/eqe.3141.

</a><font color=blue><div style="text-align: right">[up](#contents)

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
        SLF -                    	storey loss functions (linear, nonlinear or provided)
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

### Step by step procedure<a id='process'>

-> **Phase 1 - Performance objectives:**<br/>
Iterations: 1a. if EAL is not below the objective EAL limit (software outputs an Error message)
			1b. if period lower limit works out higher than the upper limit (software outputs an Error message)

		1. Define limit EAL (economic loss) and target MAFC (collapse safety) - Input
		and supply other input arguments; methods: read_inputs
		2. Supply seismic hazard and perform second-order fitting - Input, Hazard
			a. Mean annual frequency of exceeding performance limit states
			b. Peak ground acceleration at limit states relevant for EAL (e.g. SLS)
			methods: read_hazard
		3. Get the loss curve - LossCurve, EALCheck
			a. EAL calculated
			b. EAL verification
		* if EAL < EAL limit, continue, otherwise update performance objectives
		* and perform iteration 1a (phase 1)
		4. Supply storey loss functions - SLF
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
		* perform iteration 1b (phase 1 and 2)
		3. Optimization for fundamental period to identify all possible structural solutions within the period range -
		CrossSection, GetT1
			a. All solutions meeting the period range condition
			b. Optimal solution based on least weight
			
-> **Phase 3 - Collapse safety consideration:** <br/>
Iterations: 3a. Actual SPO shape differs from assumed one

		1. Identify possible static pushover curve (SPO) information as input for SPO2IDA - Input
		2. Perform SPO2IDA and derive collapse fragility function
		    	a. IDA curves
		    	b. Fractiles of R at collapse
		3. Perform optimization for MAFC - MAFCCheck
		    	a. Spectral acceleration at yield
		    	b. Yield displacement
		    	
-> **Phase 4 - Design and Detailing:**<br/>
Iterations:  4a. if local ductility requirements are not met

        	1. Identify design actions on the structure - Action
           		a. Lateral forces
        	2. Perform structural analysis and identify demands on the structural members - OpenSeesRun
		    	a. Demands
        	3. Perform moment-curvature sectional analysis to identify the necessary reinforcement, includes capacity design and local ductility requirements
			requirements per Eurocode - MPhi, Detailing
		    	a. Reinforcement ratios
		    	b. Moment capacities
		    	c. Curvature ductility
		    	d. Cracked section properties
		    	e. Hardening slope
		    	f. Softening slope and fracturing point
		    * perform iteration 4a if local ductility requirements are not met (phase 2.3 to 4)
		    * that is max reinforcement ratio is exceeded and secion redimensioning is necessitated
		    4. Estimate global hardening and fracturing ductility, peak to yield strength ratio, overstrength ratio - Detailing, Plasticity
		    * perform iteration 3a if SPO shape characteristics vary (phase 3 to 4)
		    
-> **Phase 5 - From Detailing to Global:**<br/>

        	1. Use the optimal solution and estimate Period based on cracked section properties of 4.3d - Detailing,
			CrossSection
        		a. Fundamental period
        		b. Verify that the period is within the bounds
        	2. Estimate MAFC based on the new SPO, see if an iteration is needed, establish tolerance
        	3. Check for conditions met, if non met, return to 3.1 and repeat until conditions are met - Detailing

Note: *Iterations 1a and 1b are manual, while iterations 3a and 4a may be manual or automatic.*

</a><font color=blue><div style="text-align: right">[up](#contents)
  
### Future upgrade objectives<a id='future'>

* [ ] Design of structural elements based on critical combination of M+N, M-N etc.

* [ ] Add class of reinforcement and concrete as input arguments

* [x] Add softening slope by Vecchio and Collins 1986

* [x] All input xlsx, csv etc. files will be modified to be more flexible

* [x] Add explanations on how and in which format to provide the inputs for the software

* [x] Reinforced concrete moment-resisting frame

* [x] Variable storey-loss-functions along the height

* [x] Symmetric structures only -> add considerations for 3D models

* [x] Conversion factor regression for peak floor accelerations

* [x] Code-based overstrength factors indirectly accounted for -> flexible to include

* [x] Same beam cross-sections along the height/no grouping for beams -> partially done, more will be added when necessary
 
* [ ] Considerations for shear design, will be useful also in identifying inelastic rotation capacities, then input ro_sh based on that for Haselton platic hinge length definition

* [x] SLF generator -> based on Sebastiano's work


</a><font color=blue><div style="text-align: right">[up](#contents)
  
