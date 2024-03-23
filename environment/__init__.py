from environment.cartpole_environment import CartpoleEnvironment
from environment.humaniod_standing_environment import HumaniodStandingEnvironment
from environment.bolla_rolla_environment import BollaRollaEnvironment

def create_environment(env_name):
    if env_name == "CartpoleEnvironment":
        return CartpoleEnvironment()
    elif env_name == "HumaniodStandingEnvironment":
        return HumaniodStandingEnvironment()
    elif env_name == "BollaRollaEnvironment":
        return BollaRollaEnvironment()
    else:
        raise ValueError(f"Unknown environment: {env_name}")

__all__ = [
    "CartpoleEnvironment", 
    "HumaniodStandingEnvironment", 
    "BollaRollaEnvironment",
    "create_environment"
]