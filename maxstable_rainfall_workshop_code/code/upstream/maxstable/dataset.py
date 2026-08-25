# dataset.py - Max-Stable
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import multivariate_t


def _seed_from_key(key=None, default=42):
    """Convert a JAX key/int/None into a deterministic positive R/NumPy seed."""
    if key is None:
        return int(default)
    if isinstance(key, (int, np.integer)):
        return int(key) % (2**31 - 1) or int(default)

    arr = np.asarray(key, dtype=np.uint64).reshape(-1)
    if arr.size == 0:
        return int(default)
    weights = np.arange(1, arr.size + 1, dtype=np.uint64) * np.uint64(1000003)
    seed = int(np.sum((arr + np.uint64(1)) * weights) % np.uint64(2**31 - 1))
    return seed or int(default)


try:
    ro.r("library(SpatialExtremes)")
    spatial_extremes = ro.r

except Exception as e:
    print(f"❌ R package loading failed: {e}")
    raise





import rpy2.robjects as ro
import numpy as np
import jax.numpy as jnp

ro.r("library(SpatialExtremes)")
spatial_extremes = ro.r

import numpy as np

def get_spatialextremes_coord():
    """Get coord data from SpatialExtremes package"""
    try:
        # Load coord data from SpatialExtremes package
        ro.r("""
        library(SpatialExtremes)
        data(rainfall)
        coord <- coord[,-3]
        """)
        
        # Get coord data
        coord_data = ro.r('coord')
        
        # Convert to numpy array
        with localconverter(ro.default_converter + numpy2ri.converter):
            coord_np = ro.conversion.rpy2py(coord_data)
        
    
        return coord_np
        
    except Exception as e:
        print(f"❌ Failed to get coord data: {e}")
        return None

def create_2d_grid():
    """Use coord data from SpatialExtremes package to replace original 9x9 grid"""
    return get_spatialextremes_coord()

def setup_r_simulator():
    """Set up R simulator function - only change coordinates to 4x4"""
    
    # Create 4x4 grid coordinates
    coords_4x4 = create_2d_grid()
    
    # Create coordinate matrix directly in R using string
    coords_str = ','.join(map(str, coords_4x4.flatten()))
    
    r_code = f"""
    library(SpatialExtremes)
    
    # Create 4x4 grid coordinates directly in R
    sim_coords <- matrix(c({coords_str}), nrow=79, ncol=2, byrow=TRUE)
    colnames(sim_coords) <- c('lon', 'lat')
    
    is.positive.definite <- function(m) {{
      eigenvals <- eigen(m, symmetric = TRUE, only.values = TRUE)$values
      all(eigenvals > 0)
    }}
    
    simData_spExt <- function(n, K, coord, cov11, cov12, cov22, beta.loc, beta.scale, shape) {{
      locs <- as.double(cbind(rep(1, K), coord) %*% beta.loc)
      scales <- as.double(cbind(rep(1, K), coord) %*% beta.scale)
      sim.data <- rmaxstab(n, coord, "gauss", grid = F, cov11=cov11, cov12=cov12, cov22=cov22)
      for(j in 1:n){{
        sim.data[j,] = frech2gev(sim.data[j,], loc=locs, scale=scales, shape=shape)
      }}
      return(sim.data)
    }}
    
    simData <- function(theta_row) {{
      cov11 <- theta_row[1]
      cov12 <- theta_row[2]
      cov22 <- theta_row[3]
      beta.loc <- theta_row[4:6]
      beta.scale <- theta_row[7:9]
      shape <- theta_row[10]
      
      cov_matrix <- matrix(c(cov11, cov12, cov12, cov22), ncol = 2)
      cov_matrix <- cov_matrix + diag(1e-6, nrow = 2)
      
      determinant_value <- cov11 * cov22 - cov12^2
      
      if (determinant_value <= 0) {{
        return(NA)
      }}
      
      if (!is.positive.definite(cov_matrix)) {{
        return(NA)
      }}
      
      result <- simData_spExt(47, 79, sim_coords, cov11, cov12, cov22, beta.loc, beta.scale, shape)
      return(result)
    }}
    """
    
    try:
        ro.r(r_code)
   
        return True
    except Exception as e:
        print(f"❌ R function setup failed: {e}")
        return False

def your_simulator(theta, key=None):
    """Your original simulator, just returns (47,79) instead of (47, 79)"""
    
    try:
        ro.r(f"set.seed({_seed_from_key(key)})")
        theta_list = theta.tolist() if hasattr(theta, 'tolist') else list(theta)
        ro.globalenv['theta_py'] = ro.FloatVector(theta_list)
        result = ro.r('simData(theta_py)')
        
        if result == ro.NULL:
            return jnp.full((47,79), jnp.nan)
        
        if len(result) == 1:
            try:
                if np.isnan(float(result[0])):
                    return jnp.full((47, 79), jnp.nan)
            except:
                pass
        
        result_np = np.array(result)
        
        if result_np.ndim != 2 or result_np.shape[0] != 47:
            return jnp.full((47, 79), jnp.nan)
        
        return jnp.array(result_np)
        
    except Exception as e:

        return jnp.full((47,79), jnp.nan)



# dataset.py - Max-Stable version, keeping original function names
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from scipy.stats import multivariate_t
setup_r_simulator()



def your_prior_sampler(n, key=None):
    """New prior sampling - multivariate t distribution, using pre-computed MPLE results"""
     # New version mple_theta
    mple_theta = np.array([
        322.823031, 73.8388614, 172.064508, 33.9552569, 
        0.0400072873, -0.142553094, 5.32350297, 0.0158485072, 
        -0.0307441301, 0.201344996
    ])
    
    # New version var_cov
    mple_var_cov = np.array([
        [ 3.04228527e+03,  8.90710484e+02,  1.19063612e+03, -4.70466477e+01,  4.15321651e-03,  1.95667147e-01,  3.16125119e+01, -4.82466856e-02,  7.61382778e-02,  1.93368648e+00],
        [ 8.90710484e+02,  4.61086036e+02,  3.68411267e+02, -2.81869128e+01,  2.43992879e-02,  5.16667128e-02,  2.37122050e-01,  1.47401383e-03,  1.27774749e-02,  4.68742775e-01],
        [ 1.19063612e+03,  3.68411267e+02,  8.38112730e+02, -1.09693470e+01,  2.68847884e-02, -3.13873379e-02,  1.78409922e+01,  1.92346634e-02, -9.28893171e-02,  1.17043384e+00],
        [-4.70466477e+01, -2.81869128e+01, -1.09693470e+01,  1.38012975e+02, -1.63321367e-01, -9.02681131e-02,  6.70311423e+01, -7.86018880e-02, -4.34822229e-02,  3.87398469e-03],
        [ 4.15321650e-03,  2.43992879e-02,  2.68847884e-02, -1.63321367e-01,  2.47710992e-04, -4.08688294e-05, -8.04051796e-02,  1.25576607e-04, -3.38670434e-05, -8.37362975e-07],
        [ 1.95667147e-01,  5.16667128e-02, -3.13873379e-02, -9.02681131e-02, -4.08688294e-05,  4.65966170e-04, -4.23654274e-02, -3.58327310e-05,  2.66040432e-04, -6.37519935e-06],
        [ 3.16125119e+01,  2.37122053e-01,  1.78409922e+01,  6.70311423e+01, -8.04051796e-02, -4.23654274e-02,  5.65582497e+01, -6.10300535e-02, -5.27078516e-02,  3.20609003e-02],
        [-4.82466856e-02,  1.47401382e-03,  1.92346634e-02, -7.86018880e-02,  1.25576607e-04, -3.58327310e-05, -6.10300535e-02,  9.42152674e-05, -2.06054807e-05,  9.30998570e-06],
        [ 7.61382778e-02,  1.27774749e-02, -9.28893171e-02, -4.34822229e-02, -3.38670434e-05,  2.66040432e-04, -5.27078516e-02, -2.06054807e-05,  2.64571070e-04, -9.16869502e-05],
        [ 1.93368648e+00,  4.68742775e-01,  1.17043384e+00,  3.87398469e-03, -8.37362975e-07, -6.37519935e-06,  3.20609002e-02,  9.30998570e-06, -9.16869502e-05,  2.21093272e-03]
    ])

    # Directly use pre-computed MPLE results
    theta_obs = mple_theta
    var_cov = mple_var_cov

    if theta_obs is None:
        print("❌ MPLE result is empty, cannot perform importance sampling")
        return None

    # Multivariate t distribution parameters
    df = 5
    scale_factor = 2
    scale_matrix = scale_factor * var_cov

    print(f"📊 Covariance matrix shape: {scale_matrix.shape}")
    print(f"📊 Covariance matrix range: [{np.min(scale_matrix):.3f}, {np.max(scale_matrix):.3f}]")

    # Check positive definiteness
    if not np.all(np.linalg.eigvals(scale_matrix) > 0):
        scale_matrix += 1e-6 * np.eye(scale_matrix.shape[0])

    random_seed = _seed_from_key(key)
    samples = multivariate_t.rvs(
        loc=theta_obs,
        shape=scale_matrix,
        df=df,
        size=n,
        random_state=random_seed
    )

    if samples.ndim == 1:
        samples = samples.reshape(1, -1)

    # 🔥 Simple constraint correction
    samples = np.array(samples)
    
    # Correct covariance parameters
    samples[:, 0] = np.clip(samples[:, 0], 1.0, 999.0)  # cov11 > 0
    samples[:, 2] = np.clip(samples[:, 2], 1.0, 999.0)  # cov22 > 0
    samples[:, 1] = np.clip(samples[:, 1], -299.0, 299.0)  # cov12 range
    
    # Ensure covariance matrix is positive definite: |cov12| < sqrt(cov11 * cov22)
    max_cov12 = 0.99 * np.sqrt(samples[:, 0] * samples[:, 2])
    samples[:, 1] = np.clip(samples[:, 1], -max_cov12, max_cov12)
    
    # Correct shape parameters
    samples[:, 9] = np.clip(samples[:, 9], 0.01, 0.5)  # shape > 0

    print(f"✅ Importance prior sampling of {n} samples completed")
    return jnp.array(samples)
