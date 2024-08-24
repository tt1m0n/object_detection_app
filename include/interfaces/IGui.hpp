#ifndef IGUI_HPP_
#define IGUI_HPP_

class IGui
{
    public:
        IGui() {};
        virtual ~IGui() = default;
        virtual void initialize() = 0;
        virtual void draw() = 0;
};

#endif // IGUI_HPP_