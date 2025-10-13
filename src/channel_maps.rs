/*
Copyright (C) 2010 Christian Kothe

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.
*/

use std::collections::HashMap;

pub const GRID_RES: usize = 21;

// Channel ID flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ChannelId {
    None = 0,
    FrontLeft = 1 << 1,
    FrontCenterLeft = 1 << 2,
    FrontCenter = 1 << 3,
    FrontCenterRight = 1 << 4,
    FrontRight = 1 << 5,
    SideFrontLeft = 1 << 6,
    SideFrontRight = 1 << 7,
    SideCenterLeft = 1 << 8,
    SideCenterRight = 1 << 9,
    SideBackLeft = 1 << 10,
    SideBackRight = 1 << 11,
    BackLeft = 1 << 12,
    BackCenterLeft = 1 << 13,
    BackCenter = 1 << 14,
    BackCenterRight = 1 << 15,
    BackRight = 1 << 16,
    Lfe = 1 << 31,
}

// Channel setup configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum ChannelSetup {
    FivePointOne = (ChannelId::FrontLeft as u32) | (ChannelId::FrontCenter as u32) | (ChannelId::FrontRight as u32) |
                   (ChannelId::BackLeft as u32) | (ChannelId::BackRight as u32) | (ChannelId::Lfe as u32),
    SevenPointOne = (ChannelId::FrontLeft as u32) | (ChannelId::FrontCenter as u32) | (ChannelId::FrontRight as u32) |
                    (ChannelId::SideCenterLeft as u32) | (ChannelId::SideCenterRight as u32) |
                    (ChannelId::BackLeft as u32) | (ChannelId::BackRight as u32) | (ChannelId::Lfe as u32),
}

// Channel allocation lookup table type
pub type AllocLut = Vec<Vec<Vec<f32>>>;

// Placeholder channel maps - these would be populated with actual data
lazy_static::lazy_static! {
    pub static ref CHN_ALLOC: HashMap<u32, AllocLut> = {
        let mut m = HashMap::new();
        // Initialize with empty data for now
        m.insert(ChannelSetup::FivePointOne as u32, vec![]);
        m.insert(ChannelSetup::SevenPointOne as u32, vec![]);
        m
    };
    
    pub static ref CHN_ANGLE: HashMap<u32, Vec<f32>> = {
        let mut m = HashMap::new();
        m.insert(ChannelSetup::FivePointOne as u32, vec![-27.0, 0.0, 27.0, -105.0, 105.0]);
        m.insert(ChannelSetup::SevenPointOne as u32, vec![-30.0, 0.0, 30.0, -90.0, 90.0, -135.0, 135.0]);
        m
    };
    
    pub static ref CHN_XSF: HashMap<u32, Vec<f32>> = {
        let mut m = HashMap::new();
        m.insert(ChannelSetup::FivePointOne as u32, vec![-1.0, 0.0, 1.0, -1.0, 1.0]);
        m.insert(ChannelSetup::SevenPointOne as u32, vec![-1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
        m
    };
    
    pub static ref CHN_YSF: HashMap<u32, Vec<f32>> = {
        let mut m = HashMap::new();
        m.insert(ChannelSetup::FivePointOne as u32, vec![1.0, 1.0, 1.0, -1.0, -1.0]);
        m.insert(ChannelSetup::SevenPointOne as u32, vec![1.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0]);
        m
    };
    
    pub static ref CHN_ID: HashMap<u32, Vec<ChannelId>> = {
        let mut m = HashMap::new();
        m.insert(ChannelSetup::FivePointOne as u32, vec![
            ChannelId::FrontLeft,
            ChannelId::FrontCenter,
            ChannelId::FrontRight,
            ChannelId::BackLeft,
            ChannelId::BackRight,
            ChannelId::Lfe,
        ]);
        m.insert(ChannelSetup::SevenPointOne as u32, vec![
            ChannelId::FrontLeft,
            ChannelId::FrontCenter,
            ChannelId::FrontRight,
            ChannelId::SideCenterLeft,
            ChannelId::SideCenterRight,
            ChannelId::BackLeft,
            ChannelId::BackRight,
            ChannelId::Lfe,
        ]);
        m
    };
}
