from datetime import date
import logging

logger = logging.getLogger(__name__)

def resolve_reference_date(cultivation_date: date, harvest_date: date) -> date:
    """Resolve reference date based on cultivation and harvest dates"""
    now = date.today()
    
    if cultivation_date <= now <= harvest_date:
        logger.info(f"Crop is growing (between {cultivation_date} and {harvest_date}), using today as reference: {now}")
        return now
    elif harvest_date < now:
        logger.info(f"Crop harvested (harvest was {harvest_date}), using harvest date as reference")
        return harvest_date
    else:
        raise ValueError(f"Farm cultivation has not started yet (starts {cultivation_date})")

def get_historical_dates(reference_date: date) -> list:
    """Get reference date and previous 2 years"""
    return [
        reference_date,
        reference_date.replace(year=reference_date.year - 1),
        reference_date.replace(year=reference_date.year - 2)
    ]
