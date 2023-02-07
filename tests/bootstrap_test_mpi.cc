#include "mscclpp.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

void print_usage(const char *prog)
{
  printf("usage: %s IP:PORT\n", prog);
}

int main(int argc, const char *argv[])
{
  if (argc != 2) {
    print_usage(argv[0]);
    return -1;
  }

  MPI_Init(NULL, NULL);

  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  mscclppComm_t comm;
  const char *ip_port = argv[1];
  mscclppCommInitRank(&comm, world_size, rank, ip_port);

  int *buf = (int *)calloc(world_size, sizeof(int));
  if (buf == nullptr) {
    printf("calloc failed\n");
    return -1;
  }
  buf[rank] = rank;
  mscclppResult_t res = mscclppBootStrapAllGather(comm, buf, sizeof(int));
  if (res != mscclppSuccess) {
    printf("bootstrapAllGather failed\n");
    return -1;
  }

  for (int i = 0; i < world_size; ++i) {
    if (buf[i] != i) {
      printf("wrong data: %d, expected %d\n", buf[i], i);
      return -1;
    }
  }

  res = mscclppCommDestroy(comm);
  if (res != mscclppSuccess) {
    printf("mscclppDestroy failed\n");
    return -1;
  }

  MPI_Finalize();

  printf("Succeeded! %d\n", rank);
  return 0;
}