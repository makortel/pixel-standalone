#include <cmath>
#include <algorithm>
#include <cstdio>

#include <CL/sycl.hpp>

#include "rawtodigi_oneapi.h"

namespace oneapi {
  class Packing {
  public:
    using PackedDigiType = uint32_t;

    // Constructor: pre-computes masks and shifts from field widths
     
    inline
    constexpr Packing(unsigned int row_w, unsigned int column_w,
                      unsigned int time_w, unsigned int adc_w) :
      row_width(row_w),
      column_width(column_w),
      adc_width(adc_w),
      row_shift(0),
      column_shift(row_shift + row_w),
      time_shift(column_shift + column_w),
      adc_shift(time_shift + time_w),
      row_mask(~(~0U << row_w)),
      column_mask( ~(~0U << column_w)),
      time_mask(~(~0U << time_w)),
      adc_mask(~(~0U << adc_w)),
      rowcol_mask(~(~0U << (column_w+row_w))),
      max_row(row_mask),
      max_column(column_mask),
      max_adc(adc_mask)
    { }

    uint32_t  row_width;
    uint32_t  column_width;
    uint32_t  adc_width;

    uint32_t  row_shift;
    uint32_t  column_shift;
    uint32_t  time_shift;
    uint32_t  adc_shift;

    PackedDigiType row_mask;
    PackedDigiType column_mask;
    PackedDigiType time_mask;
    PackedDigiType adc_mask;
    PackedDigiType rowcol_mask;

    uint32_t  max_row;
    uint32_t  max_column;
    uint32_t  max_adc;
  };

   
  inline
  constexpr Packing packing() {
    return Packing(11, 11, 0, 10);
  }


   
  inline
  uint32_t pack(uint32_t row, uint32_t col, uint32_t adc) {
    constexpr Packing thePacking = packing();
    adc = std::min(adc, thePacking.max_adc);

    return (row << thePacking.row_shift) |
           (col << thePacking.column_shift) |
           (adc << thePacking.adc_shift);
  }


   
  uint32_t getLink(uint32_t ww)  {
    return ((ww >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask);
  }


   
  uint32_t getRoc(uint32_t ww) {
    return ((ww >> pixelgpudetails::ROC_shift ) & pixelgpudetails::ROC_mask);
  }


   
  uint32_t getADC(uint32_t ww) {
    return ((ww >> pixelgpudetails::ADC_shift) & pixelgpudetails::ADC_mask);
  }


   
  bool isBarrel(uint32_t rawId) {
    return (1==((rawId>>25)&0x7));
  }

   
  bool rocRowColIsValid(uint32_t rocRow, uint32_t rocCol)
  {
    constexpr uint32_t numRowsInRoc = 80;
    constexpr uint32_t numColsInRoc = 52;

    /// row and collumn in ROC representation
    return ((rocRow < numRowsInRoc) & (rocCol < numColsInRoc));
  }

   
  bool dcolIsValid(uint32_t dcol, uint32_t pxid)
  {
    return ((dcol < 26) &  (2 <= pxid) & (pxid < 162));
  }

   
  pixelgpudetails::DetIdGPU getRawId(const SiPixelFedCablingMapGPU * cablingMap, uint8_t fed, uint32_t link, uint32_t roc) {
    uint32_t index = fed * pixelgpudetails::MAX_LINK * pixelgpudetails::MAX_ROC + (link-1) * pixelgpudetails::MAX_ROC + roc;
    pixelgpudetails::DetIdGPU detId = { cablingMap->RawId[index], cablingMap->rocInDet[index], cablingMap->moduleId[index] };
    return detId;
  }

   
  pixelgpudetails::Pixel frameConversion(bool bpix, int side, uint32_t layer, uint32_t rocIdInDetUnit, pixelgpudetails::Pixel local) {

    int slopeRow  = 0, slopeCol = 0;
    int rowOffset = 0, colOffset = 0;

    if (bpix) {

      if (side == -1 && layer != 1) { // -Z side: 4 non-flipped modules oriented like 'dddd', except Layer 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8-rocIdInDetUnit)*pixelgpudetails::numColsInRoc-1;
        }
        else {
          slopeRow  = -1;
          slopeCol  = 1;
          rowOffset = 2*pixelgpudetails::numRowsInRoc-1;
          colOffset = (rocIdInDetUnit-8)*pixelgpudetails::numColsInRoc;
        } // if roc
      }
      else { // +Z side: 4 non-flipped modules oriented like 'pppp', but all 8 in layer1
        if (rocIdInDetUnit < 8) {
          slopeRow  = -1;
          slopeCol  =  1;
          rowOffset = 2*pixelgpudetails::numRowsInRoc-1;
          colOffset = rocIdInDetUnit * pixelgpudetails::numColsInRoc;
        }
        else {
          slopeRow  = 1;
          slopeCol  = -1;
          rowOffset = 0;
          colOffset = (16-rocIdInDetUnit)*pixelgpudetails::numColsInRoc-1;
        }
      }

    }
    else { // fpix
      if (side==-1) { // pannel 1
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8-rocIdInDetUnit)*pixelgpudetails::numColsInRoc-1;
        }
        else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2*pixelgpudetails::numRowsInRoc-1;
          colOffset = (rocIdInDetUnit-8)*pixelgpudetails::numColsInRoc;
        }
      }
      else { // pannel 2
        if (rocIdInDetUnit < 8) {
          slopeRow = 1;
          slopeCol = -1;
          rowOffset = 0;
          colOffset = (8-rocIdInDetUnit)*pixelgpudetails::numColsInRoc-1;
        }
        else {
          slopeRow = -1;
          slopeCol = 1;
          rowOffset = 2*pixelgpudetails::numRowsInRoc-1;
          colOffset = (rocIdInDetUnit-8)*pixelgpudetails::numColsInRoc;
        }

      } // side

    }

    uint32_t gRow = rowOffset+slopeRow*local.row;
    uint32_t gCol = colOffset+slopeCol*local.col;
    //printf("Inside frameConversion row: %u, column: %u\n", gRow, gCol);
    pixelgpudetails::Pixel global = {gRow, gCol};
    return global;
  }

   
  uint8_t conversionError(uint8_t fedId, uint8_t status, bool debug, cl::sycl::stream out)
  {
    if (debug) {
      switch (status) {
      case(1) : {
        out << "Error in Fed: " << fedId << ", invalid channel Id (errorType = 35)" << cl::sycl::endl;
        break;
      }
      case(2) : {
        out << "Error in Fed: " << fedId << ", invalid ROC Id (errorType = 36)" << cl::sycl::endl;
        break;
      }
      case(3) : {
        out << "Error in Fed: " << fedId << ", invalid dcol/pixel value (errorType = 37)" << cl::sycl::endl;
        break;
      }
      case(4) : {
        out << "Error in Fed: " << fedId << ", dcol/pixel read out of order (errorType = 38)" << cl::sycl::endl;
        break;
      }
      default:
        out << "Cabling check returned unexpected result, status = " << status << cl::sycl::endl;
      };
    }

    if(status >= 1 and status <= 4) {
      return status + 34;
    }
    return 0;
  }

   
  uint32_t getErrRawID(uint8_t fedId, uint32_t errWord, uint32_t errorType, const SiPixelFedCablingMapGPU *cablingMap, bool debug, cl::sycl::stream out)
  {
    uint32_t rID = 0xffffffff;

    switch (errorType) {
      case 25 : case 30 : case 31 : case 36 : case 40 : {
        //set dummy values for cabling just to get detId from link
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        constexpr uint32_t roc  = 1;
        const uint32_t link = (errWord >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask;
        const uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if (rID_temp != 9999) rID = rID_temp;
        break;
      }
      case 29 : {
        int chanNmbr = 0;
        constexpr int DB0_shift = 0;
        constexpr int DB1_shift = DB0_shift + 1;
        constexpr int DB2_shift = DB1_shift + 1;
        constexpr int DB3_shift = DB2_shift + 1;
        constexpr int DB4_shift = DB3_shift + 1;
        constexpr uint32_t DataBit_mask = ~(~uint32_t(0) << 1);

        const int CH1 = (errWord >> DB0_shift) & DataBit_mask;
        const int CH2 = (errWord >> DB1_shift) & DataBit_mask;
        const int CH3 = (errWord >> DB2_shift) & DataBit_mask;
        const int CH4 = (errWord >> DB3_shift) & DataBit_mask;
        const int CH5 = (errWord >> DB4_shift) & DataBit_mask;
        constexpr int BLOCK_bits      = 3;
        constexpr int BLOCK_shift     = 8;
        constexpr uint32_t BLOCK_mask = ~(~uint32_t(0) << BLOCK_bits);
        const int BLOCK = (errWord >> BLOCK_shift) & BLOCK_mask;
        const int localCH = 1*CH1+2*CH2+3*CH3+4*CH4+5*CH5;
        if (BLOCK%2==0) chanNmbr=(BLOCK/2)*9+localCH;
        else chanNmbr = ((BLOCK-1)/2)*9+4+localCH;
        if ((chanNmbr < 1)||(chanNmbr > 36)) break;  // signifies unexpected result

        // set dummy values for cabling just to get detId from link if in Barrel
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        constexpr uint32_t roc  = 1;
        const uint32_t link = chanNmbr;
        const uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if(rID_temp != 9999) rID = rID_temp;
        break;
      }
      case 37 : case 38: {
        //cabling.dcol = 0;
        //cabling.pxid = 2;
        const uint32_t roc  = (errWord >> pixelgpudetails::ROC_shift) & pixelgpudetails::ROC_mask;
        const uint32_t link = (errWord >> pixelgpudetails::LINK_shift) & pixelgpudetails::LINK_mask;
        const uint32_t rID_temp = getRawId(cablingMap, fedId, link, roc).RawId;
        if(rID_temp != 9999) rID = rID_temp;
        break;
      }
      default:
        break;
    };

    return rID;
  }
  
   
  uint8_t checkROC(uint32_t errorWord, uint8_t fedId, uint32_t link, const SiPixelFedCablingMapGPU *cablingMap, bool debug, cl::sycl::stream out)
  {
    uint8_t errorType = (errorWord >> pixelgpudetails::ROC_shift) & pixelgpudetails::ERROR_mask;
    if (errorType < 25) return 0;
    bool errorFound = false;

    switch (errorType) {
      case(25) : {
        errorFound = true;
        uint32_t index = fedId * pixelgpudetails::MAX_LINK * pixelgpudetails::MAX_ROC + (link-1) * pixelgpudetails::MAX_ROC + 1;
        if (index > 1 && index <= cablingMap->size) {
          if (!(link == cablingMap->link[index] && 1 == cablingMap->roc[index])) errorFound = false;
        }
        if (debug and errorFound) out << "Invalid ROC = 25 found (errorType = 25)" << cl::sycl::endl;
        break;
      }
      case(26) : {
        if (debug) out << "Gap word found (errorType = 26)" << cl::sycl::endl;
        errorFound = true;
        break;
      }
      case(27) : {
        if (debug) out << "Dummy word found (errorType = 27)" << cl::sycl::endl;
        errorFound = true;
        break;
      }
      case(28) : {
        if (debug) out << "Error fifo nearly full (errorType = 28)" << cl::sycl::endl;
        errorFound = true;
        break;
      }
      case(29) : {
        if (debug) out << "Timeout on a channel (errorType = 29)" << cl::sycl::endl;
        if ((errorWord >> pixelgpudetails::OMIT_ERR_shift) & pixelgpudetails::OMIT_ERR_mask) {
          if (debug) out << "...first errorType=29 error, this gets masked out" << cl::sycl::endl;
        }
        errorFound = true;
        break;
      }
      case(30) : {
        if (debug) out << "TBM error trailer (errorType = 30)" << cl::sycl::endl;
        int StateMatch_bits = 4;
        int StateMatch_shift = 8;
        uint32_t StateMatch_mask = ~(~uint32_t(0) << StateMatch_bits);
        int StateMatch = (errorWord >> StateMatch_shift) & StateMatch_mask;
        if ( StateMatch != 1 && StateMatch != 8 ) {
          if (debug) out << "FED error 30 with unexpected State Bits (errorType = 30)" << cl::sycl::endl;
        }
        if (StateMatch == 1) errorType = 40; // 1=Overflow -> 40, 8=number of ROCs -> 30
        errorFound = true;
        break;
      }
      case(31) : {
        if (debug) out << "Event number error (errorType = 31)" << cl::sycl::endl;
        errorFound = true;
        break;
      }
      default:
        errorFound = false;
    };

    return errorFound ? errorType : 0;
  }

  
  void rawtodigi_kernel(const Input *input, Output *output,
      bool useQualityInfo, bool includeErrors, bool debug, cl::sycl::nd_item<3> item_ct1, cl::sycl::stream out)
  {
    const SiPixelFedCablingMapGPU* cablingMap = &input->cablingMap;
    const uint32_t wordCounter = input->wordCounter;
    const uint32_t* word = input->word;
    const uint8_t* fedIds =input->fedId;
    uint16_t* xx = output->xx;
    uint16_t* yy = output->yy;
    uint16_t* adc = output->adc;
    uint32_t* pdigi = output->digi;
    uint32_t* rawIdArr = output->rawIdArr;
    uint16_t* moduleId = output->moduleInd;
    GPU::SimpleVector<PixelErrorCompact>* err = &output->err;


    int32_t first = item_ct1.get_local_id(2) + item_ct1.get_group(2)*item_ct1.get_local_range().get(2);
    for (int32_t iloop=first, nend=wordCounter; iloop<nend; iloop+=item_ct1.get_local_range().get(2)*item_ct1.get_group_range(2)) { 

        auto gIndex  = iloop;
        xx[gIndex]   = 0;
        yy[gIndex]   = 0;
        adc[gIndex]  = 0;
        bool skipROC = false;

        uint8_t fedId = fedIds[gIndex/2]; // +1200;

        // initialize (too many coninue below)
        pdigi[gIndex]  = 0;
        rawIdArr[gIndex] = 0;
        moduleId[gIndex] = 9999;

        uint32_t ww = word[gIndex]; // Array containing 32 bit raw data
        if (ww == 0) {
          // 0 is an indicator of a noise/dead channel, skip these pixels during clusterization
          continue;
        }

        uint32_t link  = getLink(ww);            // Extract link
        uint32_t roc   = getRoc(ww);             // Extract Roc in link
        pixelgpudetails::DetIdGPU detId = getRawId(cablingMap, fedId, link, roc);

        uint8_t errorType = checkROC(ww, fedId, link, cablingMap, debug, out);
        skipROC = (roc < pixelgpudetails::maxROCIndex) ? false : (errorType != 0);
        if (includeErrors and skipROC)
        {
          uint32_t rID = getErrRawID(fedId, ww, errorType, cablingMap, debug, out);
          err->push_back(PixelErrorCompact{rID, ww, errorType, fedId});
          continue;
        }

        uint32_t rawId  = detId.RawId;
        uint32_t rocIdInDetUnit = detId.rocInDet;
        bool barrel = isBarrel(rawId);

        uint32_t index = fedId * pixelgpudetails::MAX_LINK * pixelgpudetails::MAX_ROC + (link-1) * pixelgpudetails::MAX_ROC + roc;
        if (useQualityInfo) {
          skipROC = cablingMap->badRocs[index];
          if (skipROC) continue;
        }

        uint32_t layer = 0;//, ladder =0;
        int side = 0, panel = 0, module = 0;//disk = 0, blade = 0

        if (barrel)
        {
          layer  = (rawId >> pixelgpudetails::layerStartBit) & pixelgpudetails::layerMask;
          module = (rawId >> pixelgpudetails::moduleStartBit) & pixelgpudetails::moduleMask;
          side   = (module < 5)? -1 : 1;
        }
        else {
          // endcap ids
          layer = 0;
          panel = (rawId >> pixelgpudetails::panelStartBit) & pixelgpudetails::panelMask;
          //disk  = (rawId >> diskStartBit_) & diskMask_;
          side  = (panel == 1)? -1 : 1;
          //blade = (rawId >> bladeStartBit_) & bladeMask_;
        }

        // ***special case of layer to 1 be handled here
        pixelgpudetails::Pixel localPix;
        if (layer == 1) {
          uint32_t col = (ww >> pixelgpudetails::COL_shift) & pixelgpudetails::COL_mask;
          uint32_t row = (ww >> pixelgpudetails::ROW_shift) & pixelgpudetails::ROW_mask;
          localPix.row = row;
          localPix.col = col;
          if (includeErrors) {
            if (not rocRowColIsValid(row, col)) {
              uint8_t error = conversionError(fedId, 3, debug, out);  // use the device function and fill the arrays
              err->push_back(PixelErrorCompact{rawId, ww, error, fedId});
              if(debug) out << "TODO - output needs update" << cl::sycl::endl;
              continue;
            }
          }
        } else {
          // ***conversion rules for dcol and pxid
          uint32_t dcol = (ww >> pixelgpudetails::DCOL_shift) & pixelgpudetails::DCOL_mask;
          uint32_t pxid = (ww >> pixelgpudetails::PXID_shift) & pixelgpudetails::PXID_mask;
          uint32_t row  = pixelgpudetails::numRowsInRoc - pxid/2;
          uint32_t col  = dcol*2 + pxid%2;
          localPix.row = row;
          localPix.col = col;
          if (includeErrors and not dcolIsValid(dcol, pxid)) {
            uint8_t error = conversionError(fedId, 3, debug, out);
            err->push_back(PixelErrorCompact{rawId, ww, error, fedId});
            if(debug) out << "TODO - output needs update" << cl::sycl::endl;
            continue;
          }
        }

        pixelgpudetails::Pixel globalPix = frameConversion(barrel, side, layer, rocIdInDetUnit, localPix);
        xx[gIndex]    = globalPix.row;  // origin shifting by 1 0-159
        yy[gIndex]    = globalPix.col;  // origin shifting by 1 0-415
        adc[gIndex]   = getADC(ww);
        pdigi[gIndex] = pack(globalPix.row, globalPix.col, adc[gIndex]);
        moduleId[gIndex] = detId.moduleId;
        rawIdArr[gIndex] = rawId;
    } // end of loop (gIndex < end)

  } // end of Raw to Digi kernel

  void rawtodigi(const Input *input_d, Output *output_d,
                 const uint32_t wordCounter,
                 bool useQualityInfo, bool includeErrors, bool debug, cl::sycl::queue & queue) try {
    const int threadsPerBlock = 512;
    const int blocks = (wordCounter + threadsPerBlock-1) /threadsPerBlock; // fill it all

    {
      queue.submit(
        [&](cl::sycl::handler &cgh) {
          cl::sycl::stream out(64 * 1024, 80, cgh);
          auto dpct_global_range = cl::sycl::range<3>(blocks, 1, 1) * cl::sycl::range<3>(threadsPerBlock, 1, 1);
          auto dpct_local_range = cl::sycl::range<3>(threadsPerBlock, 1, 1);
          cgh.parallel_for<dpct_kernel_name<class rawtodigi_kernel_5b0d6a>>(
            cl::sycl::nd_range<3>(cl::sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1), dpct_global_range.get(0)), cl::sycl::range<3>(dpct_local_range.get(2), dpct_local_range.get(1), dpct_local_range.get(0))),
            [=](cl::sycl::nd_item<3> item_ct1) {
              rawtodigi_kernel(input_d, output_d, useQualityInfo, includeErrors, debug, item_ct1, out);
            });
        });
    }
  }
catch (cl::sycl::exception const &exc) {
  std::cerr << exc.what() << "EOE at line " << __LINE__ << std::endl;
  std::exit(1);
}     
  
} // end namespace oneapi
