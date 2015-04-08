#include "copyHelpers.h"
#include <memory>
#include "amp.h"
#include "cl_manage.h"

// dst: Destination memory address 
// dst_offset: The offset where to begin copying data from dst. If dst is host buffer, the offset
//                  is always 0
// src: Source memory address 
// dst_offset: The offset where to begin copying data from src. If dst is host buffer, the offset
//                  is always 0
// count: Size in bytes to copy 
// kind: Type of transfer
int gpuMemcpy(void* dst, size_t dst_offset, void* src, size_t src_offset,
              size_t count, gpuMemcpyKind kind)
{
  switch(kind)
  {
    case gpuMemcpyHostToHost:
      memcpy(dst, src, count);
      break;

    case gpuMemcpyDeviceToHost: {
      cl_int err = clEnqueueReadBuffer(Concurrency::getOCLQueue(src),
                                       static_cast<cl_mem>(src), CL_TRUE, src_offset,
                                       count, dst, 0, NULL, NULL);

      if (err != CL_SUCCESS)
      {
        printf("Read error = %d\n", err);
        exit(1);
      }

      break;
    }

    case gpuMemcpyHostToDevice: {
      cl_int err = clEnqueueWriteBuffer(Concurrency::getOCLQueue(dst),
                                        static_cast<cl_mem>(dst), CL_TRUE, dst_offset,
                                        count, src, 0, NULL, NULL);

      if (err != CL_SUCCESS)
      {
        printf("Write error = %d\n", err);
        exit(1);
      }

      break;
    }

    case gpuMemcpyDeviceToDevice: {
      cl_command_queue srcq = Concurrency::getOCLQueue(src);
      cl_command_queue dstq = Concurrency::getOCLQueue(dst);
      // TODO: since there is only one queue on one device now
      if (srcq == dstq) {
        cl_event event;
        cl_int err = clEnqueueCopyBuffer (Concurrency::getOCLQueue(src),
                                        static_cast<cl_mem>(src), static_cast<cl_mem>(dst),
                                        src_offset, dst_offset, count, 0, NULL, &event);

        if (err != CL_SUCCESS)
        {
          printf("Copy error = %d\n", err);
          exit(1);
        }

        clWaitForEvents(1, &event);
        break;
      } else {
        // peer to peer with host as bridge
        // TODO: replace with peer2peer copying
        float *host = (float *) malloc(count);
        assert (host);
        gpuMemcpy(host, 0, src, src_offset, count, gpuMemcpyDeviceToHost);
        gpuMemcpy(dst, dst_offset, host, 0, count, gpuMemcpyHostToDevice);
        free (host);
        host = 0;
      }
    }

    case gpuMemcpyDefault:
      break;
  }
  return 0;
}

// FIXME: is it correct?
int gpuMemcpyAsync(void* dst, size_t dst_offset, void* src, size_t src_offset,
                   size_t count, gpuMemcpyKind kind)
{
  switch(kind)
  {
    case gpuMemcpyHostToHost:
      memcpy(dst, src, count);
      break;

    case gpuMemcpyDeviceToHost: {
      cl_event event;
      cl_int err = clEnqueueWriteBuffer(Concurrency::getOCLQueue(src),
                                        static_cast<cl_mem>(src), CL_FALSE, src_offset,
                                        count, dst, 0, NULL, &event);

      if (err != CL_SUCCESS)
      {
        printf("WriteSync error = %d\n", err);
        exit(1);
      }

      cl_int status = CL_QUEUED;
      while(status != CL_COMPLETE)
      {
        clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                       sizeof(cl_int), &status, NULL);
      }

      break;
    }

    case gpuMemcpyHostToDevice: {
      cl_event event;
      cl_int err = clEnqueueReadBuffer(Concurrency::getOCLQueue(dst),
                                       static_cast<cl_mem>(dst), CL_FALSE, dst_offset,
                                       count, src, 0, NULL, &event);

      if (err != CL_SUCCESS)
      {
        printf("ReadSync error = %d\n", err);
        exit(1);
      }

      cl_int status = CL_QUEUED;
      while(status != CL_COMPLETE)
      {
        clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                       sizeof(cl_int), &status, NULL);
      }

      break;
    }

    case gpuMemcpyDeviceToDevice: {
      cl_command_queue srcq = Concurrency::getOCLQueue(src);
      cl_command_queue dstq = Concurrency::getOCLQueue(dst);
      // TODO: since there is only one queue on one device now
      if (srcq == dstq) {
        cl_event event;
        cl_int err = clEnqueueCopyBuffer (Concurrency::getOCLQueue(src),
                                        static_cast<cl_mem>(src), static_cast<cl_mem>(dst),
                                        src_offset, dst_offset, count, 0, NULL, &event);

        if (err != CL_SUCCESS)
        {
          printf("CopyAsync error = %d\n", err);
          exit(1);
        }

        cl_int status = CL_QUEUED;
        while(status != CL_COMPLETE)
        {
          clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                             sizeof(cl_int), &status, NULL);
        }

        break;
      }else {
        // peer to peer with host as bridge
        // TODO: replace with peer2peer copying
        float *host = (float *) malloc(count);
        assert (host);
        gpuMemcpyAsync(host, 0, src, src_offset, count, gpuMemcpyDeviceToHost);
        gpuMemcpyAsync(dst, dst_offset, host, 0, count, gpuMemcpyHostToDevice);
        free (host);
        host = 0;
      }
    }

    case gpuMemcpyDefault:
      break;
  }
  return 0;
}
