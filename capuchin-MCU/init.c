/*
 * init.c
 * This file contains functions which initialize the device.
 * Include initialization of MSP pins, UART configuration, and the clock system.
 */

#include "init.h"

void init_gpio(void) {
    /**
     * Initializes all pins to output and sets pins to LOW. This
     * prevents unnecessary current consumption by floating pins.
     */
    P1DIR = 0xFF;
    P1OUT = 0x0;
    P2DIR = 0xFF;
    P2OUT = 0x0;
    P3DIR = 0xFF;
    P3OUT = 0x0;
    P4DIR = 0xFF;
    P4OUT = 0x0;
    P5DIR = 0xFF;
    P5OUT = 0x0;
    P6DIR = 0xFF;
    P6OUT = 0x0;
    P7DIR = 0xFF;
    P7OUT = 0x0;
    P8DIR = 0xFF;
    P8OUT = 0x0;
    PADIR = 0xFF;
    PAOUT = 0x0;
    PBDIR = 0xFF;
    PBOUT = 0x0;
    PCDIR = 0xFF;
    PCOUT = 0x0;
    PDDIR = 0xFF;
    PDOUT = 0x0;

    PM5CTL0 &= ~LOCKLPM5;
}


void init_clock_system(void) {
    /*
     * set up system clock
     */

    // Configure one FRAM waitstate as required by the device datasheet for MCLK
    // operation beyond 8MHz _before_ configuring the clock system.
    FRCTL0 = FRCTLPW | NWAITS_1;

    // Clock System Setup - 16MHz
    // followed this post: https://e2e.ti.com/support/microcontrollers/msp-low-power-microcontrollers-group/msp430/f/msp-low-power-microcontroller-forum/609971/msp430fr5994-can-t-get-system-to-work-on-16-mhz
    // and this TI's official example code: https://dev.ti.com/tirex/explore/node?node=ALMKUpgS2sr.Sf-qEyGcAQ__IOGqZri__LATEST
    /* ACLK = ~9.4kHz,  SMCLK = MCLK = 16MHz */
    CSCTL0_H = CSKEY_H;                     // Unlock CS registers
    CSCTL1 = DCOFSEL_0;                     // Set DCO to 1MHz
    // Set SMCLK = MCLK = DCO, ACLK = VLOCLK
    CSCTL2 = SELA__VLOCLK | SELS__DCOCLK | SELM__DCOCLK;
    // Per Device Errata set divider to 4 before changing frequency to
    // prevent out of spec operation from overshoot transient
    CSCTL3 = DIVA__4 | DIVS__4 | DIVM__4;   // Set all corresponding clk sources to divide by 4 for errata
    CSCTL1 = DCOFSEL_4 | DCORSEL;           // Set DCO to 16MHz
    // Delay by ~10us to let DCO settle. 60 cycles = 20 cycles buffer + (10us / (1/4MHz))
    __delay_cycles(60);
    CSCTL3 = DIVA__1 | DIVS__1 | DIVM__1;   // Set all dividers to 1 for 16MHz operation
    CSCTL0_H = 0;                           // Lock CS registers                      // Lock CS registers

}
