#include <cuda_runtime_api.h>
#include <string>

namespace cuda {

/* \brief Number of accessible devices */
int num_devices();

int current_device();

int sm_arch(int device_id = -1);

/* \brief Number of multiprocessors on a device
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \return Number of multiprocessors
 */
int sm_count(int device_id = -1);

}  // namespace cuda