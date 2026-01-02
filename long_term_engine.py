from models import Pilot, YearGroup, Qual
import pandas as pd # Useful for the line graph later
from logic import TrainingEnvironment

class TrainingEnvironment:
    @staticmethod
    def get_annual_aging_rates(exp_ratio: float, paa: int, ute: float):
        """
        Inputs: current state of the CAF
        Outputs: rates for the next 12 months
        """
        absorption_modifier = 1.0 if exp_ratio >= 0.45 else (exp_ratio / 0.45)
        
        annual_sorties = (paa * ute * 12) * absorption_modifier
        annual_hours = annual_sorties * 1.3  
        
        return int(annual_sorties), int(annual_hours)

class CAFSimulation:
    def __init__(self):
        self.pilots: List[Pilot] = []
        self.history = []

    def run_year(self, year: int, intake_size: int, paa: int, ute: float, retention: float):
        # Calculate current Experience Ratio (ER)
        active = [p for p in self.pilots if p.active]
        num_exp = sum(1 for p in active if p.qual in [Qual.FL, Qual.IP])
        er = num_exp / len(active) if active else 0.5 # Default to healthy if empty

        # GET AGING RATES from our lookup logic
        rates = TrainingEnvironment.calculate_annual_rates(paa, ute, er)

        # 1. Add new year group pilots
        for _ in range(intake_size):
            self.pilots.append(Pilot(year_group=year, adsc_remaining=10))

        # 2. Age existing pilots
        stats = {'Year': year, 'WG': 0, 'FL': 0, 'IP': 0, 'ER': er, 'Sorties': rates['sorties']}
        
        for p in active:
            # Pass the averaged rates into the pilot's aging function
            p.age_one_year_with_rates(rates['sorties'], rates['hours'])
            
            # Apply your retention logic
            if p.adsc_remaining <= 0 and random.random() > retention:
                p.separate()
            
            if p.active:
                stats[p.qual.name] += 1
        
        self.history.append(stats)

    def run_year(self, current_sim_year: int, intake_size: int, retention_rate: float):
        # 1. Calculate the current Absorption Ratio before aging
        # This represents the training environment the pilots "lived" through this year
        total_wgs = sum(1 for yg in self.year_groups for p in yg.pilots if p.active and p.qual == Qual.WG)
        total_leads = sum(1 for yg in self.year_groups for p in yg.pilots if p.active and p.qual != Qual.WG)
        
        # Logic: If ratio < 1.0 (more WGs than Leads), flying hours drop.
        # We'll use 1.5 as the "ideal" ratio for full 100% sortie production.
        ratio = total_leads / max(total_wgs, 1)
        current_sortie_rate = min(120, int(100 * (ratio / 1.5))) 
        current_hour_rate = int(current_sortie_rate * 1.3) 

        # 2. Add new year group (The "Accession" class)
        self.add_year_group(current_sim_year, intake_size, initial_adsc=10)

        # 3. Age and track stats
        # We initialize counters for the history log
        stats = {
            'Year': current_sim_year, 
            'WG': 0, 
            'FL': 0, 
            'IP': 0, 
            'Total_Active': 0,
            'Sortie_Rate': current_sortie_rate # Useful to track how "healthy" training was
        }
        
        for yg in self.year_groups:
            for p in yg.pilots:
                if p.active:
                    # Age the pilot based on the current year's absorption capacity
                    p.age_one_year(current_sortie_rate, current_hour_rate)
                    
                    # Check for retention (if ADSC expired)
                    p.check_retention(retention_rate)
                    
                    # Re-check if still active after retention roll
                    if p.active:
                        stats[p.qual.name] += 1
                        stats['Total_Active'] += 1

        # 4. Append to history so we can graph it later
        self.history.append(stats)

    def get_dataframe(self):
        return pd.DataFrame(self.history)