Step-by-step procedure for Integrated Performance-Based Seismic Design

Developed by D.Shahnazaryan

Step vs relevant classes
"""
Sample:
1. Step - Class
	a. Outputs
"""

Input arguments are:    1. Limit EAL                Limiting economic loss value in terms of expected annual loss
                        2. Target MAFC              Target collapse safety value in terms of mean annual frequency of
                                                    exceedance
                        3. input.csv                input file comprising of the following arguments:
                            design_scenario: tag/id of case to be used to store results in \database
                            PLS: performance limit states of interest
                            ELR: expected loss ratios corresponding to PLS
                            TR: return periods corresponding to PLS
                            aleatory: aleatory uncertainties corresponding to PLS
                            SLF: storey loss functions (linear, nonlinear or provided)
                            bldg_ch: building characteristics (floor loading in kPa, roof loading in kPa, floor area)
                            h_storeys: heights of stories
                            mode_red: higher mode reduction factor
                            PFA_convert: peak floor acceleration conversion factor
                            spans_X: bay widths
                        4. slf.csv                  Storey-loss-functions file:

                        5. Hazard file

Classes ready: external/HazardFit, client/Hazard, client/ErrorCheck,

--> Functionality of a software
1. Input:
        a. Reading of the input data and storing the information; Input - read_inputs
        b. Reading of hazard input; Input - read_hazard
        c. Fitting the hazard if necessary - HazardFit
4.




Time stamps as relative to each other, since for different input arguments it may vary.

-> Phase 1 - Performance objectives: 0.0 seconds, 0.0 minutes
		1. Define limit EAL (economic loss) and target MAFC (collapse safety) - Input
		and supply other input arguments
		    methods: read_inputs
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
-> Phase 2 - Building information: 3.6 seconds, 0.06 minutes
		1. Get design spectra at limit states relevant for EAL (e.g. SLS) - Spectra
			a. Sa and Sd
		2. Get feasible period range - PeriodRange
			a. Lower period bound
			b. Upper period bound
		3. Optimization for fundamental period to identify all possible structural solutions within the period range -
		CrossSection, GetT1
			a. All solutions meeting the period range condition
			b. Optimal solution based on least weight
-> Phase 3 - Collapse safety consideration: 1.1 seconds, 0.01 minutes
		1. Identify possible static pushover curve (SPO) information as input for SPO2IDA - Input
		2. Perform SPO2IDA and derive collapse fragility function
		    a. IDA curves
		    b. Fractiles of R at collapse
		3. Perform optimization for MAFC - MAFCCheck
            a. Spectral acceleration at yield
            b. Yield displacement
-> Phase 4 - Design: 8.2 seconds, 0.13 minutes
        1. Identify design actions on the structure - Action
            a. Lateral forces
        2. Perform ELFM and identify demands on the structural members - OpenSeesRun
            a. Demands
        3. Perform moment-curvature sectional analysis to identify the necessary reinforcement, includes capacity design
           requirements per Eurocode - MPhi, Detailing
            a. Reinforcement ratios
            b. Moment capacities
            c. Curvature ductility
            d. Cracked section properties
-> Phase 5 - Detailing:
        1. Use the optimal solution and estimate Period based on cracked section properties of 4.3d - Detailing, CrossSection
        	a. Fundamental period
        	b. Verify that the period is within the bounds
        2. Estimate system hardening ductility - Detailing
        	a. System hardening ductility
        3. todo, add estimations of other parameters of SPO curve
        X. Check for conditions met, if non met, return to 3.1 and repeat until conditions are met - Detailing
-> Phase 6 - Modifications and Rerunning of Phases if necessary:
        1.



-> Current version main limitations and assumptions (further development necessary) vs possible solutions:
    * All input xlsx, csv etc. files will be modified to be more flexible
	* IMPORTANT: FOR KWARGS function refer to get_emf function
	* Add explanations on how and in which format to provide the inputs for the software
	* Reinforced concrete moment-resisting frame -> Steel MRF, other typologies to be included
	* 3 Performance objective limit states -> flexible for inclusion
	* Homogeneous storey-loss-functions along the height -> flexible for inclusion
	* SPO2IDA tool for collapse fragility definition -> ML algorithms to avoid using SPO2IDA, needs extensive studies and analysis, possible data collection
	* Symmetric structures only -> add considerations for 3D models
	* Single conversion factor for peak floor accelerations -> study based on Antonio's work to include regressions
	* Code-based overstrength factors indirectly accounted for -> flexible to include
	* Performs equivalent lateral force method of analysis for element design -> RMSA not really necessary, if assymmetric, perform MA, then combine for RMSA, add separate Gravity only analysis, add gravity + ELFM analysis
	* Same beam cross-sections along the height/no grouping for beams -> add grouping once generic opensees model is updated to account for it
	* Library, frameworks for GUI version - https://blog.resellerclub.com/the-6-best-python-gui-frameworks-for-developers/
	* Considerations for shear design, will be useful also in identifying inelastic rotation capacities
	* Haselton - softening slope definition implementation via post capping rotation capacity
	