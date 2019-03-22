#ifndef PIXELGPUDETAILS_H
#define PIXELGPUDETAILS_H

namespace pixelgpudetails {
  // Maximum fed for phase1 is 150 but not all of them are filled
  // Update the number FED based on maximum fed found in the cabling map
  constexpr unsigned int MAX_FED  = 150;
  constexpr unsigned int MAX_LINK =  48;  // maximum links/channels for Phase 1
  constexpr unsigned int MAX_ROC  =   8;
  constexpr unsigned int MAX_SIZE = MAX_FED * MAX_LINK * MAX_ROC;

  // Phase 1 geometry constants
  constexpr uint32_t layerStartBit    = 20;
  constexpr uint32_t ladderStartBit   = 12;
  constexpr uint32_t moduleStartBit   = 2;

  constexpr uint32_t panelStartBit    = 10;
  constexpr uint32_t diskStartBit     = 18;
  constexpr uint32_t bladeStartBit    = 12;

  constexpr uint32_t layerMask        = 0xF;
  constexpr uint32_t ladderMask       = 0xFF;
  constexpr uint32_t moduleMask       = 0x3FF;
  constexpr uint32_t panelMask        = 0x3;
  constexpr uint32_t diskMask         = 0xF;
  constexpr uint32_t bladeMask        = 0x3F;

  constexpr uint32_t LINK_bits        = 6;
  constexpr uint32_t ROC_bits         = 5;
  constexpr uint32_t DCOL_bits        = 5;
  constexpr uint32_t PXID_bits        = 8;
  constexpr uint32_t ADC_bits         = 8;

  // special for layer 1
  constexpr uint32_t LINK_bits_l1     = 6;
  constexpr uint32_t ROC_bits_l1      = 5;
  constexpr uint32_t COL_bits_l1      = 6;
  constexpr uint32_t ROW_bits_l1      = 7;
  constexpr uint32_t OMIT_ERR_bits    = 1;

  constexpr uint32_t maxROCIndex      = 8;
  constexpr uint32_t numRowsInRoc     = 80;
  constexpr uint32_t numColsInRoc     = 52;

  constexpr uint32_t MAX_WORD = 2000;
  constexpr uint32_t MAX_FED_WORDS   = MAX_FED * MAX_WORD;

  constexpr uint32_t ADC_shift  = 0;
  constexpr uint32_t PXID_shift = ADC_shift + ADC_bits;
  constexpr uint32_t DCOL_shift = PXID_shift + PXID_bits;
  constexpr uint32_t ROC_shift  = DCOL_shift + DCOL_bits;
  constexpr uint32_t LINK_shift = ROC_shift + ROC_bits_l1;
  // special for layer 1 ROC
  constexpr uint32_t ROW_shift = ADC_shift + ADC_bits;
  constexpr uint32_t COL_shift = ROW_shift + ROW_bits_l1;
  constexpr uint32_t OMIT_ERR_shift = 20;

  constexpr uint32_t LINK_mask = ~(~uint32_t(0) << LINK_bits_l1);
  constexpr uint32_t ROC_mask  = ~(~uint32_t(0) << ROC_bits_l1);
  constexpr uint32_t COL_mask  = ~(~uint32_t(0) << COL_bits_l1);
  constexpr uint32_t ROW_mask  = ~(~uint32_t(0) << ROW_bits_l1);
  constexpr uint32_t DCOL_mask = ~(~uint32_t(0) << DCOL_bits);
  constexpr uint32_t PXID_mask = ~(~uint32_t(0) << PXID_bits);
  constexpr uint32_t ADC_mask  = ~(~uint32_t(0) << ADC_bits);
  constexpr uint32_t ERROR_mask = ~(~uint32_t(0) << ROC_bits_l1);
  constexpr uint32_t OMIT_ERR_mask = ~(~uint32_t(0) << OMIT_ERR_bits);

  struct DetIdGPU {
    uint32_t RawId;
    uint32_t rocInDet;
    uint32_t moduleId;
  };

  struct Pixel {
   uint32_t row;
   uint32_t col;
  };
}

struct alignas(128) SiPixelFedCablingMapGPU {
  unsigned int fed[pixelgpudetails::MAX_SIZE];
  unsigned int link[pixelgpudetails::MAX_SIZE];
  unsigned int roc[pixelgpudetails::MAX_SIZE];
  unsigned int RawId[pixelgpudetails::MAX_SIZE];
  unsigned int rocInDet[pixelgpudetails::MAX_SIZE];
  unsigned int moduleId[pixelgpudetails::MAX_SIZE];
  unsigned char badRocs[pixelgpudetails::MAX_SIZE];
  unsigned int size = 0;
};

struct PixelErrorCompact {
  uint32_t rawId;
  uint32_t word;
  unsigned char errorType;
  unsigned char fedId;
};

#endif
