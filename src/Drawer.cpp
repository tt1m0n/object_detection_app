#include "Drawer.hpp"

Drawer::Drawer(std::unique_ptr<IGui> gui) : gui(std::move(gui))
{
}

void Drawer::run()
{
    gui->initialize();
    gui->draw();
}