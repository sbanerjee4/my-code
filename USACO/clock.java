import java.util.*;
import java.io.*;
import java.lang.*;
public class clock
{
   static ArrayList<String> num = new ArrayList<>();
   public static void main(String[] args) throws FileNotFoundException
   {
      Scanner in = new Scanner(new File("clock.in"));
      int[] clocks = new int[9];
      for(int i = 0; i < 9; i++)
         clocks[i] = (in.nextInt() / 3) % 4;
      
      numGen("", 1);
      for(String s : num)
      {
         boolean works = isok(s, clocks);
         if(works)
         {
            boolean printed = false;
            for(int i = 0; i < s.length(); i++)
            {
               if(s.charAt(i) != '0')
               {
                  if(!printed)
                     System.out.print(s.charAt(i));
                  else System.out.print(" " + s.charAt(i));
                  printed = true;
               }
            }
            System.out.println();
            System.exit(0);
         }
      }
   }
   
   public static void numGen(String s, int move)
   {
      if(move > 10) 
         return;
      if(move == 10)
         num.add(s);
        
      numGen(s + "0", move + 1);
      numGen(s + move, move + 1);
      numGen(s + move + "" + move, move + 1);
      numGen(s + move + "" + move + move, move + 1);
   }
   
   public static boolean isok(String s, int[] temp)
   {
      int[] clocks = new int[9];
      for(int i = 0; i < 9; i++)
         clocks[i] = temp[i];
        
        
      for(char c : s.toCharArray())
      {
         if(c == '1')
         {
            clocks[0]++;
            clocks[1]++;
            clocks[3]++;
            clocks[4]++;
         }
         else if(c == '2')
         {
            clocks[0]++;
            clocks[1]++;
            clocks[2]++;
         }
         else if(c == '3')
         {
            clocks[1]++;
            clocks[2]++;
            clocks[4]++;
            clocks[5]++;
         }
         else if(c == '4')
         {
            clocks[0]++;
            clocks[3]++;
            clocks[6]++;
         }
         else if(c == '5')
         {
            clocks[1]++;
            clocks[3]++;
            clocks[4]++;
            clocks[5]++;
            clocks[7]++;
         }
         else if(c == '6')
         {
            clocks[2]++;
            clocks[5]++;
            clocks[8]++;
         }
         else if(c == '7')
         {
            clocks[3]++;
            clocks[4]++;
            clocks[6]++;
            clocks[7]++;
         }
         else if(c == '8')
         {
            clocks[6]++;
            clocks[7]++;
            clocks[8]++;
         }
         else if(c == '9')
         {
            clocks[4]++;
            clocks[5]++;
            clocks[7]++;
            clocks[8]++;
         }
      }
        
      for(int i : clocks)
         if(i % 4 != 0)
            return false;
      return true;
   }
}