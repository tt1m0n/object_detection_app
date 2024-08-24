#ifndef DRAWER_HPP
#define DRAWER_HPP

#include <memory>
#include "IGui.hpp"

class Drawer
{
    public:
        Drawer(std::unique_ptr<IGui> gui);
        void run();
    
    private:
        std::unique_ptr<IGui> gui;
};

#endif // DRAWER_HPP